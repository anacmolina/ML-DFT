#from ase.parallel import parprint as print

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

#from ase.parallel import parprint as print

import os
import pickle

import ase
import chemcoord as cc
from gpaw import GPAW
from ase import Atoms
from ase.io import read # Check if it's need it
from ase.optimize import BFGS
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.io import Trajectory

#import gpaw.mpi as mpi
#rank = mpi.world.rank

# Angles mapping (Object) Makes mapping (tangent or arctangent) to the tensor that receives
class Angles_mapping:
    
    def __init__(self, n=5):
        self.n = n
    
    def tensor_checking(self, tensor):
        if torch.is_tensor(tensor):
             pass
        else:
             raise RuntimeError("It must be a tensor")
        
        if len(tensor.shape)==2:
            pass
        else:
            raise RuntimeError("Shape not accepted")

    def mapping(self, tensor):
        self.tensor_checking(tensor)
        tensor[:, self.n:] = tensor[:, self.n:].tan()
        
    def inv_mapping(self, tensor):
        self.tensor_checking(tensor)
        tensor[:, self.n:] = tensor[:, self.n:].arctan()
   
# Construction table for both isomers, pandas dataframe (convention we chose)
def get_construction_table():

   index = np.append(0, np.append(np.arange(2,6), 1))
   construction_table = pd.DataFrame(index=index)
      
   construction_table['b'] = ['origin', 0, 2, 2, 4, 4]
   construction_table['a'] = ['e_z', 'e_z', 0, 3, 2, 2]
   construction_table['d'] = ['e_x', 'e_x', 'e_x', 0, 3, 3]
      
   return construction_table        
         
# Structure: Class that uses construction table and symbols to build a molecules
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
         
         b[0] = 12.649508829797915
         a[0:2] = np.array([0.88610283, 1.64261783])
         d[0:3] = np.array([ 0.61541089, -1.94064131, -0.73241593])
         
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
      
   def calculate_potential_energy(self, zmat_values_=None):
      
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

      cell = [16, 16, 16]
      self.molecule.set_cell(cell)
      self.molecule.center()
      self.molecule.set_pbc(True)
      
      # DFT calculator low level precision but faster (takes 1 minute in serial)                                \
         
      self.calculator = GPAW(mode = 'lcao', basis='pvalence.dz', h =0.2, xc = 'PBE', spinpol = True, nbands = -4\
                             , txt='ag6.out')

      #DFT calculator with higher precision but takes longer (about 30 minutes in serial).                      \
         
      #calc = GPAW(mode = 'fd', h =0.18, xc = 'PBE', eigensolver = 'rmm-diis', spinpol = True, nbands=-4)       \
         
      self.molecule.set_calculator(self.calculator)
      
      # Calculating the potential energy
      self.potential_energy = self.molecule.get_potential_energy()
   
# Getting the collective variables from internal coordinates
def get_CVs(data):
   #symbols = np.full(6, 'Ag')
   C_vals = []
   R_vals = []
   for x in data:
      ag6 = Structure()
      ag6.build_zmat_matrix(x)
      atoms = ag6.molecule                                                                                  
      C_vals.append(C(atoms))
      R_vals.append(R(atoms))
   return C_vals, R_vals

d=2.8
rij_d = lambda rij: rij/d

def X_i(i, r):
   value=0
   for j in range(len(r)):
      if(i!=j):
         value = value + (1 - rij_d(r[i][j])**8)/(1 - rij_d(r[i][j])**16)
   return value

# Coordination number  
def C(atoms):
   r = atoms.get_all_distances()
   return np.array([X_i(i, r) for i in range(atoms.get_global_number_of_atoms())]).sum()

# Radius of gyration
def R(atoms):
   r_rcm = atoms.get_positions() - atoms.get_center_of_mass()
   result = np.sqrt(np.array([np.linalg.norm(ri)**2 for ri in r_rcm]).sum()/atoms.get_global_number_of_atoms())
   return result         

# Plotting FES and database
def plotting_fes_db(train_data=None):
   from flonacomldft.FES.plotter2 import Plotter
   from flonacomldft.md_utils import get_path

   ceph_home = get_path()
   
   plotting = Plotter(400, 'Ag6')
   plotting.readfile(ceph_home + 'unrotated_300.txt')
   
   ax = plotting.plot_fes(0.1, 300, delta2=1, shift=1.5)
   
   if train_data is not None:

      db1 = pd.read_csv(ceph_home + 'is1_lcao_zmat.csv')
      db1 = db1.drop(['energies'], axis=1)
      db1 = db1.to_numpy()
   
      c_db1, r_db1 = get_CVs(db1)

      db2 = pd.read_csv(ceph_home + 'is2_lcao_zmat.csv')
      db2 = db2.drop(['energies'], axis=1)
      db2 = db2.to_numpy()
   
      c_db1, r_db1 = get_CVs(db1)
      c_db2, r_db2 = get_CVs(db2)
   
      ax.plot(c_db1, r_db1, 'c.', label='DB is1')
      ax.plot(c_db2, r_db2, 'c.', label='DB is2')
   
   return ax

# Plot a molecule 2D (Plot with structure NOT center!)    
def plot_sample(x):
   from ase.visualize.plot import plot_atoms
   fig, ax = plt.subplots()
   symbols = np.full(6, 'Ag')
   ag6 = Structure()
   ag6.build_zmat_matrix(x)
   plot_atoms(ag6.molecule, ax)

def shuffle_arr(vs, indexes):
    concat = lambda vs: torch.cat(vs)
    v = concat(vs)
    return v[indexes]

def rephase(zmat, angle=0, columns=['dihedral13']):
    for column in columns:
        phase = np.zeros(zmat[column].shape)
        phase[zmat[column]>angle] = -2*np.pi
        zmat[column] = zmat[column] + phase
    return zmat

def deg_to_rad(zmat):
    labels = zmat.columns.to_list()
    for label in labels[6:-1]:
        zmat[label] = np.deg2rad(zmat[label].tolist())
    return zmat

def get_internal_coordinates(traj):

    traj = traj
    construction_table = get_construction_table()
    energies = [traj_.get_potential_energy() for traj_ in traj]
    
    xyz = []
    for traj_ in traj:
        xyz.append(cc.Cartesian.from_ase_atoms(traj_))
    
    zmat = [xyz_.get_zmat(construction_table) for xyz_ in xyz]
    
    b = construction_table.b.to_numpy()
    a = construction_table.a.to_numpy()
    d = construction_table.d.to_numpy()
    ind = construction_table.index.to_numpy()

    label_b = ['bond'+str(i)+str(j) for i, j in zip(ind, b)]
    label_a = ['angle'+str(i)+str(j) for i, j in zip(ind, a)]
    label_d = ['dihedral'+str(i)+str(j) for i, j in zip(ind, d)]

    cols = label_b + label_a + label_d + ['energies']
    
    new_zmat = pd.DataFrame(columns = cols, index=np.arange(0, len(zmat), 1))
    
    for i in range(len(zmat)):
        new_zmat.iloc[i] = zmat[i].iloc[:, 2].tolist()+zmat[i].iloc[:, 4].tolist()+zmat[i].iloc[:, 6].tolist()+[energies[i]]
    
    new_zmat = deg_to_rad(new_zmat)
    new_zmat = rephase(new_zmat)

    new_zmat = new_zmat.drop(["bond0origin", "angle0e_z", "angle2e_z", "dihedral0e_x", "dihedral2e_x", "dihedral3e_x"], axis=1)

    new_zmat = new_zmat.to_numpy(dtype=np.float32)

    return torch.from_numpy(new_zmat).float()

# Molecular structures, minimums

def get_is1():
    pos = np.array([[7.9804600, 5.464791, 8.0],
                    [7.9611420, 10.16485, 8.0],
                    [6.5837500, 7.868667, 8.0],
                    [5.2993890, 5.513795, 8.0],
                    [9.3697860, 7.876240, 8.0],
                    [10.665305, 5.524828, 8.0]])
    isomer = Atoms('Ag6', positions=pos)
    return isomer

def get_is2():
    pos = np.array([[6.5910080, 5.595878, 7.139020],
                    [9.4089920, 5.595878, 7.139020],
                    [5.7157570, 8.272920, 7.161092],
                    [8.0000000, 7.529808, 8.358414],
                    [10.284243, 8.272920, 7.161092],
                    [8.0000000, 9.919833, 7.129457]])
    isomer = Atoms('Ag6', positions=pos)
    return isomer

# Running MD
def run_molecular_dynamics(molecule, iters, name):
    import gpaw.mpi as mpi
    rank = mpi.world.rank

    ceph_home = get_path()
    mol = molecule

    #Setting the cell

    mol.set_cell([16, 16, 16])
    mol.set_pbc(True)
    mol.center()

    # Building calculator
    calc = GPAW(mode="lcao", h=0.2, basis="pvalence.dz", spinpol=True, xc="PBE", symmetry="off", nbands = -4, txt='ag6_md_'+name+'.out')

    mol.set_calculator(calc)

    # Adding conditions to the MD simulation
    MaxwellBoltzmannDistribution(mol, temperature_K=300)
    Stationary(mol)
    ZeroRotation(mol)
    
    file = ceph_home+'ag6_'+name+'.traj'

    mpi.world.barrier()

    # Running the MD
    dyn = NVTBerendsen(mol, 5 * units.fs, taut = 50, temperature_K=300, trajectory=file)
    dyn.run(iters)
    
    # Getting the MD trajectory
    mpi.world.barrier()
    traj = Trajectory(file)

    return traj