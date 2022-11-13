import os
from tkinter import NONE
from flonacomldft.utils.data_utils import trajectories_folder

import numpy as np
import torch

import chemcoord as cc
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.io import Trajectory
#from ase.parallel import parprint as print

from flonacomldft.internal_coordinates import get_internal_coordinates
from flonacomldft.utils.silver_isomers_utils import get_construction_table

from gpaw import GPAW
import gpaw.mpi as mpi

# Structure: Class that uses construction table and symbols to build a molecules
#add option to save .gpw file
#reuse the calculator for the next energy calculation
class Structure:
   """
   Structure (Object)

   zmat ---> xyz

   get_potential_energy()
   """

   def __init__(self, construction_table_=get_construction_table(), 
                symbols_=np.full(6, 'Ag'), txt='ag6.out'):
      super().__init__()
      
      self.construction_table = construction_table_.copy()
      self.symbols = symbols_
      self.Natoms = len(self.symbols)
      
      # DFT calculator low level precision but faster (takes 1 minute in serial)                                \
      self.calculator = GPAW(mode = 'lcao', basis='pvalence.dz', h =0.2, xc = 'PBE',
                             spinpol = True, nbands = -4, txt=txt)

      #DFT calculator with higher precision but takes longer (about 30 minutes in serial).                      \
         
      #calc = GPAW(mode = 'fd', h =0.18, xc = 'PBE', eigensolver = 'rmm-diis', spinpol = True, nbands=-4) 
      
   def build_zmat_matrix_and_molecule(self, zmat_values):  
      """"
      Build the zmat matrix and the molecule from the zmat values
      
      In:
         zmat_values: array with the values of the internal coordinates
      Out:  
         zmat_matrix: df-zmat of coordinates in the basis of the IC
         molecule: Ase Atoms object 
               - storing cartesian coordinates, forces, energies, etc.
      
      """

      if torch.is_tensor(zmat_values):
         self.zmat_values = zmat_values.clone().detach().numpy()
      else:
         self.zmat_values = zmat_values.copy()
               
      zmat_matrix = self.construction_table.copy()
      
      b = np.zeros(6)
      a = np.zeros(6)
      d = np.zeros(6)
      
      if len(self.zmat_values)==12:
         
         # reference frame shift - values taken from chemcoord
         b[0] = 1.27
         a[0:2] = np.array([2.21657, 2.21657])
         d[0:3] = np.array([2.21657, 2.21657, 2.21657])

         b[1:] = self.zmat_values[:5]
         a[2:] = self.zmat_values[5:9]
         d[3:] = self.zmat_values[9:]
         
         a = np.rad2deg(a)
         d = np.rad2deg(d)
         
         b = np.double(b)
         a = np.double(a)
         d = np.double(d)
         
         zmat_matrix.insert(0, "atom", self.symbols, True)
         zmat_matrix.insert(2, "bond", b, True)
         zmat_matrix.insert(4, "angle", a, True)
         zmat_matrix.insert(6, "dihedral", d, True)

         zmat_matrix = cc.Zmat(zmat_matrix)
         molecule = self.zmat_matrix.get_cartesian().get_ase_atoms()

      else:
         raise RuntimeError('Data not valid')
      
      return zmat_matrix, molecule 
      
   def calculate_potential_energy(self, zmat_values):
      
      dim = zmat_values.shape[0]
      zmat_matrix, molecule = self.build_zmat_matrix_and_molecule(zmat_values)

      # Setting the cell parameters
      cell = [16, 16, 16]
      molecule.set_cell(cell)
      molecule.center()
      molecule.set_pbc(True)
      molecule.set_calculator(self.calculator)
      
      # Calculating the potential energy
      self.potential_energy = molecule.get_potential_energy()

# Running MD
def run_molecular_dynamics(molecule, iters, name, starting=True):
    rank = mpi.world.rank

    # Setting a folder to save the trajectories
    path = os.path.join(os.getcwd(), 'trajectories')
    #mpi.world.barrier()
   
    if rank==0 and os.path.isdir(path)==False:
       print('Folder created: %s Rank: %d '%(path, rank))
       os.mkdir(path)
    else:
       pass
    
    mpi.world.barrier()

    #Setting the cell
    molecule.set_cell([16, 16, 16])
    molecule.set_pbc(True)
    molecule.center()

    file = path+'/ag6_'+name

    # Building calculator
    calc = GPAW(mode="lcao", h=0.2, basis="pvalence.dz", spinpol=True, xc="PBE", symmetry="off", nbands = -4, txt=file+'.out')

    molecule.set_calculator(calc)
   
    if starting==True:
       # Adding conditions to the MD simulation
       MaxwellBoltzmannDistribution(molecule, temperature_K=300)
       Stationary(molecule)
       ZeroRotation(molecule)
    else:
       pass

    mpi.world.barrier()

    # Running the MD
    dyn = NVTBerendsen(molecule, 5 * units.fs, taut = 50, temperature_K=300,
                       trajectory=file+'.traj')
    dyn.run(iters)
    
    # Getting the MD trajectory
    mpi.world.barrier()
    traj = Trajectory(file+'.traj')

    return traj

def run_md_get_zmat(molecule, iterations, file_name, starting=True):
    traj = run_molecular_dynamics(molecule, iterations, file_name, starting)
    zmat = get_internal_coordinates(traj).detach()
    return zmat