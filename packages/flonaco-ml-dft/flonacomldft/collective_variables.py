# TODO: Clean up
# TODO: Try to vectorize this code

# libraries

import numpy as np
#from flonacomldft.internal_coordinates import Coordinates_mapping

### Define collective variables constants    
d=2.8
rij_d = lambda rij: rij/d

### Define collective variables function: Switching function
def X_i(i, r):
   
   value=0
   
   for j in range(len(r)):
   
      if(i!=j):
   
         value = value + (1 - rij_d(r[i][j])**8)/(1 - rij_d(r[i][j])**16)
   
   return value

### Define collective variables function: Coordination number
def compute_C(atoms):
   
   r = atoms.get_all_distances()
   
   return np.array([X_i(i, r) for i in range(atoms.get_global_number_of_atoms())]).sum()

### Define collective variables function: Radius of gyration
def compute_R(atoms):
   
   r_rcm = atoms.get_positions() - atoms.get_center_of_mass()
   result = np.sqrt(np.array([np.linalg.norm(ri)**2 for ri in r_rcm]).sum()/atoms.get_global_number_of_atoms())
   
   return result 

def get_collective_variables(atoms):
    """Get the collective variables of the system."""
    return  np.array((compute_C(atoms), compute_R(atoms)))

### Define collective variables function: Get collective variables from internal coordinates
#TODO: Use internal coordinates to compute collective variables
#def get_CVs(data):
#   
#   C_vals = []
#   R_vals = []
#   
#   for x in data:
#   
#      coord_maps = Coordinates_mapping()
#      molecule = coord_maps.build_molecule_from_zmat(x)
#                                                                               
#      C_vals.append(compute_C(molecule))
#      R_vals.append(compute_R(molecule))
#   
#   return C_vals, R_vals

def get_cvs_from_traj(traj):
    """Get the collective variables from a trajectory."""
    return np.array([get_collective_variables(atoms) for atoms in traj])

#def get_cvs_from_zmat(data):
#
#   cvs = []
#
#   for x in data:
#         
#      coord_maps = Coordinates_mapping()
#      molecule = coord_maps.build_molecule_from_zmat(x)
#
#      cvs.append(get_collective_variables(molecule))
#
#   return np.array(cvs)
#
#def get_cvs_from_real_centered(data, isomer):
#
#   cvs = []
#
#   for x in data:
#         
#      coord_maps = Coordinates_mapping()
#      molecule = coord_maps.build_molecule_from_real_centered(x.reshape(1, -1), isomer)
#
#      cvs.append(get_collective_variables(molecule[0]))
#
#   return np.array(cvs)