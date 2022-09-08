import torch
import numpy as np
from flonacomldft.dft_utils import Structure

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

# Getting the collective variables from internal coordinates
def get_CVs(data):
   
   C_vals = []
   R_vals = []
   
   for x in data:
   
      ag6 = Structure()
      ag6.build_zmat_matrix(x)
   
      atoms = ag6.molecule                                                                                  
   
      C_vals.append(C(atoms))
      R_vals.append(R(atoms))
   
   return C_vals, R_vals