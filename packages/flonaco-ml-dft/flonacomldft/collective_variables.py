import torch
import numpy as np
from flonacomldft.internal_coordinates import Structure

d=2.8
rij_d = lambda rij: rij/d

def X_i(i, r):
   
   value=0
   
   for j in range(len(r)):
   
      if(i!=j):
   
         value = value + (1 - rij_d(r[i][j])**8)/(1 - rij_d(r[i][j])**16)
   
   return value

def compute_C(atoms):
   
   r = atoms.get_all_distances()
   
   return np.array([X_i(i, r) for i in range(atoms.get_global_number_of_atoms())]).sum()

def compute_R(atoms):
   
   r_rcm = atoms.get_positions() - atoms.get_center_of_mass()
   result = np.sqrt(np.array([np.linalg.norm(ri)**2 for ri in r_rcm]).sum()/atoms.get_global_number_of_atoms())
   
   return result 

def get_CVs(data):
   
   C_vals = []
   R_vals = []
   
   for x in data:
   
      structure = Structure()
      molecule = structure.build_molecule(x)
                                                                               
      C_vals.append(compute_C(molecule))
      R_vals.append(compute_R(molecule))
   
   return C_vals, R_vals