import torch
import numpy as np
import pandas as pd
import chemcoord as cc

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
    
    try:
        energies = [traj_.get_potential_energy() for traj_ in traj]
    except:
        energies = [0]*len(traj)
        #raise RuntimeWarning("No calculator")

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

def get_pos_energy(zmat):
    u_tensor = zmat[:, -1]
    x_tensor = zmat[:, :-1]
    return x_tensor, u_tensor

def get_mix_data(data_1, data_2):
    xi_is1, ui_is1 = get_pos_energy(data_1)
    xi_is2, ui_is2 = get_pos_energy(data_2)
    ci_is1, ci_is2 = torch.zeros(xi_is1.shape[0]), torch.ones(xi_is2.shape[0]) 

    n_points = xi_is1.shape[0] + xi_is2.shape[0]

    indexes = torch.randperm(n_points)

    # Unifying all data from MD and shuffling in order to to Metropolis-Hastings
    xis = shuffle_arr([xi_is1, xi_is2], indexes)
    uis = shuffle_arr([ui_is1, ui_is2], indexes)
    cis = shuffle_arr([ci_is1, ci_is2], indexes)
    return xis, uis, cis