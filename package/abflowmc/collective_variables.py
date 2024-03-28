# libraries
import numpy as np

#collective variables parameters
d=2.8

# collective variables function: rij/d for coordination number
rij_d = lambda rij: rij/d

# collective variables function: Switching function for coordination number
def X_i(i, r):
   """
   Switching function for the coordination number
   Args:
         i: index of the atom
         r: array of distances between the atoms with i-th atom
   Returns:
         The switching function for the coordination number
   """
   value=0
   
   for j in range(len(r)):
   
      if(i!=j):
   
         value = value + (1 - rij_d(r[i][j])**8)/(1 - rij_d(r[i][j])**16)
   
   return value

# collective variables function: Coordination number
def compute_C(atoms):
   """
   Compute the coordination number of the system
   Args:
         atoms: ASE atoms object
   Returns:
         The coordination number of the system
   """
   
   r = atoms.get_all_distances()
   
   return np.array([X_i(i, r) for i in range(atoms.get_global_number_of_atoms())]).sum()

# collective variable function: Radius of gyration
def compute_R(atoms):
   """
   Compute the radius of gyration of the system
   Args:
         atoms: ASE atoms object
   Returns:
         The radius of gyration of the system
   """
   r_rcm = atoms.get_positions() - atoms.get_center_of_mass()
   result = np.sqrt(np.array([np.linalg.norm(ri)**2 for ri in r_rcm]).sum()/atoms.get_global_number_of_atoms())
   
   return result 

def get_collective_variables(atoms):
   """Get the collective variables of the system
   Args:
      atoms: ASE atoms object
   Returns:
      The collective variables of the system
   """
   return  np.array((compute_C(atoms), compute_R(atoms)))


def get_cvs_from_traj(traj):
   """Get the collective variables from a trajectory
   Args:
      traj: ASE trajectory object
   Returns:
      The collective variables of the trajectory
   """
   return np.array([get_collective_variables(atoms) for atoms in traj])

