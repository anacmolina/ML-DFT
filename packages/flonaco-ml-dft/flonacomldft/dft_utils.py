import os
import pickle
from tkinter import NONE
from flonacomldft.data_utils import trajectories_folder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

#from ase.parallel import parprint as print

import ase
import chemcoord as cc
from gpaw import GPAW
from ase import Atoms
from ase.optimize import BFGS
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.io import Trajectory

from flonacomldft.internal_coordinates import get_construction_table


class Angles_transformation(torch.Tensor):
   def __init__(self, x_):
      super().__init__()

      if torch.is_tensor(x_):
         pass
      else:
         raise RuntimeError("It must be a tensor")

      self.x = x_

      print(self.x, self.x.shape)
      if(len(self.x.shape)==1):
         self.x = self.x.reshape(1, 12)

      print(self.x, self.x.shape)

      self.dims = self.x.shape[1]
      self.Nsample = self.x.shape[0]

      print(self.dims, self.Nsample)   
      if self.dims==12:
         self.n = 5
      else:
         raise RuntimeError('Can not define transformation')

   def transf(self):
         self.x[:,self.n:] = torch.tan(self.x[:,self.n:])
         #self.x = self.x.reshape(self.Nsample, self.dims)

   def inv_transf(self):
         self.x[:,self.n:] = torch.arctan(self.x[:,self.n:])
         #self.x = self.x.reshape(self.Nsample, self.dims)

       
"""

Structure (Object)

zmat ---> xyz
xyz <--- zmat

get_potential_energy()
"""

# Structure: Class that uses construction table and symbols to build a molecules

#add option to save .gpw file
#reuse the calculator for the next energy calculation
class Structure:
   def __init__(self, construction_table_=get_construction_table(), symbols_=np.full(6, 'Ag')):
      super().__init__()
      
      self.construction_table = construction_table_.copy()
      self.symbols = symbols_
      self.Natoms = len(self.symbols)
      
      self.zmat_values = None
      self.zmat_matrix = None
      self.molecule = None
      self.potential_energy = None
      self.calculator = None
      
   def build_zmat_matrix(self, zmat_values_):

      if(zmat_values_ is None):
         
         if(self.zmat_values is None):
            raise RuntimeError('No data')
      else:
         if torch.is_tensor(zmat_values_):
            self.zmat_values = zmat_values_.clone()
            self.zmat_values = self.zmat_values.detach().numpy()
         else:
            self.zmat_values = zmat_values_.copy()
               
      zmat_matrix = self.construction_table.copy()
      
      b = np.zeros(6)
      a = np.zeros(6)
      d = np.zeros(6)
      
      if len(self.zmat_values)==12:
         
         b[0] = 12.649508829797915 #12.551959 #
         a[0:2] = np.array([0.88610283, 1.64261783])
         d[0:3] = np.array([0.61541089, -1.94064131, -0.73241593])
         
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

         self.zmat_matrix = cc.Zmat(zmat_matrix)
         self.molecule = self.zmat_matrix.get_cartesian().get_ase_atoms()

      else:
         raise RuntimeError('Data not valid')
      
   def calculate_potential_energy(self, zmat_values_=None, txt='ag6.out'):
      
      if (zmat_values_ is None and self.zmat_values is not None):
         dim = self.zmat_values.shape[0]
         self.build_zmat_matrix(self.zmat_values)
         
      elif (zmat_values_ is not None):
         self.zmat_values = zmat_values_
         dim = self.zmat_values.shape[0]
         self.build_zmat_matrix(zmat_values_)
         
      elif (zmat_values_ is None and self.zmat_values is None):
         raise RuntimeError('No data')
      
      # Setting the cell parameters

      from ase.visualize import view

      cell = [16, 16, 16]
      self.molecule.set_cell(cell)
      self.molecule.center()
      self.molecule.set_pbc(True)

      view(self.molecule)
      
      # DFT calculator low level precision but faster (takes 1 minute in serial)                                \
         
      self.calculator = GPAW(mode = 'lcao', basis='pvalence.dz', h =0.2, xc = 'PBE', spinpol = True, nbands = -4\
                             , txt=txt)

      #DFT calculator with higher precision but takes longer (about 30 minutes in serial).                      \
         
      #calc = GPAW(mode = 'fd', h =0.18, xc = 'PBE', eigensolver = 'rmm-diis', spinpol = True, nbands=-4)       \
         
      self.molecule.set_calculator(self.calculator)
      
      # Calculating the potential energy
      self.potential_energy = self.molecule.get_potential_energy()

# Running MD
def run_molecular_dynamics(molecule, iters, name, starting=True):
    #from flonacomldft.data_utils import trajectories_folder
    import gpaw.mpi as mpi
    rank = mpi.world.rank

    path = os.path.join(os.getcwd(), 'trajectories')
    mpi.world.barrier()
   
    if rank==0 and os.path.isdir(path)==False:
       print('Folder created: %s Rank: %d '%(path, rank))
       os.mkdir(path)
    else:
       pass
    
    mpi.world.barrier()
    mol = molecule

    #Setting the cell

    mol.set_cell([16, 16, 16])
    mol.set_pbc(True)
    mol.center()

    file = path+'/ag6_'+name

    # Building calculator
    calc = GPAW(mode="lcao", h=0.2, basis="pvalence.dz", spinpol=True, xc="PBE", symmetry="off", nbands = -4, txt=file+'.out')

    mol.set_calculator(calc)
   
    if starting==True:
       # Adding conditions to the MD simulation
       MaxwellBoltzmannDistribution(mol, temperature_K=300)
       Stationary(mol)
       ZeroRotation(mol)
    else:
       pass

    mpi.world.barrier()

    # Running the MD
    dyn = NVTBerendsen(mol, 5 * units.fs, taut = 50, temperature_K=300, trajectory=file+'.traj')
    dyn.run(iters)
    
    # Getting the MD trajectory
    mpi.world.barrier()
    traj = Trajectory(file+'.traj')

    return traj

def compute_energy(zmat, txt='ag6.out'):
   ag6 =  Structure()
   print('here')
   ag6.calculate_potential_energy(zmat, txt=txt)
   U = ag6.potential_energy
   return U