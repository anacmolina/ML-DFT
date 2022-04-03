import chemcoord as cc
import torch
from gpaw import GPAW
import numpy as np
import pandas as pd
import ase
import sys
import os

from ase.parallel import parprint as print
from ase.visualize.plot import plot_atoms

from flonacomldft.FES.plotter2 import Plotter

# Get path to database
def get_path():
   if os.path.isdir('/mnt/home/amolina/ceph/database/'):
      ceph_home = '/mnt/home/amolina/ceph/database/'
   elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'):
      ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'
   elif os.path.isdir('/home/anacristina/ml_dft_project/database/'):
      ceph_home = '/home/anacristina/ml_dft_project/database/'
   elif os.path.isdir('/home/amolina/ml_dft_project/database/'):
      ceph_home = '/home/amolina/ml_dft_project/database/'
   else:
      raise RuntimeError('Data path not understood')
   return ceph_home

# Transformation for angles (torch.tensor) 
class Angles_transformation(torch.Tensor):
    def __init__(self, x_):
        super().__init__()
        
        if torch.is_tensor(x_):
            pass
        else:
            raise RuntimeError("It must be a tensor.")
        
        self.x = x_
        self.n = None
        self.dims = len(x_.shape)
        
    def transf(self, n_):
        self.n = n_
        if self.dims==1:
            self.x[self.n:] = torch.tan(self.x[self.n:])
        else:
            self.x[:,self.n:] = torch.tan(self.x[:,self.n:])
    
    def inv_transf(self, n_):
        self.n = n_       
        if self.dims==1:
            self.x[self.n:] = torch.arctan(self.x[self.n:])
        else:
            self.x[:,self.n:] = torch.arctan(self.x[:,self.n:])

# Calculating the energy (pd.DataFrame, np.array or list, int)
class Structure:
    def __init__(self, construction_table_, symbols_, Natoms_):
        super().__init__()
        self.zmat_values = None
        self.construction_table = construction_table_.copy()
        self.symbols = symbols_
        self.Natoms = Natoms_
        self.zmat_matrix = None
        self.molecule = None
        self.potential_energy = None
        self.calculator = None

        # (np.array (add tensor type))
    def build_zmat_matrix(self, zmat_values_):
        self.zmat_values = zmat_values_.copy()
        
        if(self.zmat_values is None):
            raise RuntimeError('No data')

        zmat_array = zmat_values_.copy()
        angles = np.rad2deg(zmat_array[6:])
        zmat_values = np.append(zmat_array[:6], angles).reshape(3, self.Natoms)
        zmat_values = zmat_values.astype('float64')
        
        zmat_matrix = self.construction_table.copy()
    
        zmat_matrix.insert(0, "atom", self.symbols, True)  
        zmat_matrix.insert(2, "bond", zmat_values[0], True)
        zmat_matrix.insert(4, "angle", zmat_values[1], True)
        zmat_matrix.insert(6, "dihedral", zmat_values[2], True)
        
        self.zmat_matrix = cc.Zmat(zmat_matrix)
        self.molecule = self.zmat_matrix.get_cartesian().get_ase_atoms()

    def build_zmat_matrix_rd(self, zmat_values_):
        z_values = zmat_values_.copy()

        if(z_values is None):
            raise RuntimeError('No data')

        zmat_matrix = self.construction_table.copy()
    
        b = np.zeros(6)
        a = np.zeros(6)
        d = np.zeros(6)
    
        b[1:] = z_values[:5]
        a[1:] = z_values[5:10]
        d[1:] = z_values[10:]
    
        a = np.rad2deg(a)
        d = np.rad2deg(d)

        zmat_matrix.insert(0, "atom", self.symbols, True)
        zmat_matrix.insert(2, "bond", b, True)
        zmat_matrix.insert(4, "angle", a, True)
        zmat_matrix.insert(6, "dihedral", d, True)
    
        self.zmat_matrix = cc.Zmat(zmat_matrix)
        self.molecule = self.zmat_matrix.get_cartesian().get_ase_atoms()    
        
        # (np.array (add tensor type))
    def calculate_potential_energy(self, zmat_values_=None):

        if(zmat_values_ is None and self.zmat_values is not None):
           dim = self.zmat_values.shape[0]
        elif(zmat_values_ is not None):
           self.zmat_values = zmat_values_
           dim = self.zmat_values.shape[0]
        elif(zmat_values_ is None and self.zmat_values is None):
           raise RuntimeError('No data')
        
        if dim == 18: 
           self.build_zmat_matrix(zmat_values_)
        else:
           self.build_zmat_matrix_rd(zmat_values_)
        
        cell = [16, 16, 16]
        self.molecule.set_cell(cell)
        self.molecule.center()
        self.molecule.set_pbc(True)

        # DFT calculator low level precision but faster (takes 1 minute in serial)
        self.calculator = GPAW(mode = 'lcao', basis='pvalence.dz', h =0.2, xc = 'PBE', spinpol = True, nbands = -4, txt='ag6.out')

        #DFT calculator with higher precision but takes longer (about 30 minutes in serial).
        #calc = GPAW(mode = 'fd', h =0.18, xc = 'PBE', eigensolver = 'rmm-diis', spinpol = True, nbands=-4)

        self.molecule.set_calculator(self.calculator)
        
        self.potential_energy = self.molecule.get_potential_energy()
"""
    def calculate_potential_energy_parallel(self, zmat_values_, comm):

        import gpaw.mpi as mpi

        # DFT calculator low level precision but faster (takes 1 minute in serial)                                
        self.calculator = GPAW(mode = 'lcao', basis='pvalence.dz', h =0.2, xc = 'PBE', spinpol = True, nbands = -4, communicator = comm, txt='ag6.out')

        #DFT calculator with higher precision but takes longer (about 30 minutes in serial).                      
	#calc = GPAW(mode = 'fd', h =0.18, xc = 'PBE', eigensolver = 'rmm-diis', spinpol = True, nbands=-4)  
        
        self.build_zmat_matrix(zmat_values_)
        
        cell = [16, 16, 16]
        self.molecule.set_cell(cell)
        self.molecule.center()
        self.molecule.set_pbc(True)

        
        self.molecule.set_calculator(self.calculator)
        
        self.potential_energy = self.molecule.get_potential_energy()
"""
        
# Getting the construction table for each isomer        
def AG6_construction_tables(isomer_):

    if (isomer_=='is1'):

        index = np.append(0, np.append(np.arange(2,6), 1))
        construction_table_is1 = pd.DataFrame(index=index)

        construction_table_is1['b'] = ['origin', 0, 2, 2, 4, 4]
        construction_table_is1['a'] = ['e_z', 'e_z', 0, 3, 2, 2]
        construction_table_is1['d'] = ['e_x', 'e_x', 'e_x', 0, 3, 3]
        
        return(construction_table_is1)
    
    elif (isomer_=='is2'):
        
        index = np.append(3, np.append(np.arange(0,3), [4, 5]))
        construction_table_is2 = pd.DataFrame(index=index)

        construction_table_is2['b'] = ['origin', 3, 0, 0, 1, 4]
        construction_table_is2['a'] = ['e_z', 'e_z', 3, 1, 0, 1]
        construction_table_is2['d'] = ['e_x', 'e_x', 'e_x', 3, 2, 0]
        
        return(construction_table_is2)

    else:
        
        raise RuntimeError('The AG6 isomer can not be recognized')
    
# Getting the collective variables

def get_CVs(data, ct):
    symbols = np.full(6, 'Ag')
    ct_ = ct
    C_vals = []
    R_vals = []
    dim = data.shape[1]
    for x in data:
        ag6 = Structure(construction_table_=ct_, symbols_=symbols, Natoms_=len(symbols))
        if dim==18:
           ag6.build_zmat_matrix(x)
        else:
           ag6.build_zmat_matrix_rd(x)
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

def C(atoms):
    r = atoms.get_all_distances()
    return np.array([X_i(i, r) for i in range(atoms.get_global_number_of_atoms())]).sum()

def R(atoms):
    r_rcm = atoms.get_positions() - atoms.get_center_of_mass()
    result = np.sqrt(np.array([np.linalg.norm(ri)**2 for ri in r_rcm]).sum()/atoms.get_global_number_of_atoms())
    return result         

#Plotting FES and database

def plotting_fes_db(isomer_, mode_, ct):

    ceph_home = get_path()
   
    isomer = str(isomer_)
    mode = str(mode_)
    name=isomer+'_'+mode
    
    plotting = Plotter(400, 'Ag6')
    plotting.readfile(ceph_home + 'unrotated_300.txt')
    
    ax = plotting.plot_fes(0.1, 300, delta2=0.1, shift=1)
    
    db = pd.read_csv(ceph_home + name +'_zmat.csv')
    db = db.drop(['energies'], axis=1)
    db = db.to_numpy()
   
    c_db, r_db = get_CVs(db, ct)

    ax.plot(c_db, r_db, 'c.', label='Database')
    
    return ax
 
# Plot a molecule 2D (MISSING! Center the structure!)    
def plot_sample(x, name, ct):
    fig, ax = plt.subplots()
    symbols = np.full(6, 'Ag')
    ct_ = ct
    ag6 = Structure(construction_table_=ct_, symbols_=symbols, Natoms_=len(symbols))
    ag6.build_zmat_matrix(x)
    plot_atoms(ag6.molecule, ax)
    fig.savefig('ag6_'+name+'.png')
