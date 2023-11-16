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


def get_cvs_from_traj(traj):
    """Get the collective variables from a trajectory."""
    return np.array([get_collective_variables(atoms) for atoms in traj])
