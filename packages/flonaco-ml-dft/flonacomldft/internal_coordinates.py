import torch
import numpy as np
import pandas as pd
import chemcoord as cc

from flonacomldft.utils.silver_isomers_utils import get_construction_table

class Angles_mapping:
    def __init__(self, idx_first_angle=5):
        self.idx_first_angle = idx_first_angle
    
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
        tensor[:, self.idx_first_angle:] = tensor[:, self.idx_first_angle:].tan()
        
    def inv_mapping(self, tensor):
        self.tensor_checking(tensor)
        tensor[:, self.idx_first_angle:] = tensor[:, self.idx_first_angle:].arctan()

def rephase(zmat, angle=0, columns=['dihedral13']):
    for column in columns:
        phase = np.zeros(zmat[column].shape)
        phase[zmat[column]>angle] = -2 * np.pi
        zmat[column] = zmat[column] + phase
    return zmat

def deg_to_rad(zmat, labels):
    #labels = zmat.columns.to_list()
    #for label in labels[6:-1]:
    for label in labels:
        zmat[label] = np.deg2rad(zmat[label].tolist())
    return zmat

def get_internal_coordinates(traj):
    """"
    traj - What type of objects can traj be? Ase-atoms?
    """
    construction_table = get_construction_table()
    
    try:
        energies = [traj_.get_potential_energy() for traj_ in traj]
        ENERGY = True
    except:
        ENERGY = False

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

    if ENERGY: cols = label_b + label_a + label_d + ['energies']
    else: cols = label_b + label_a + label_d

    new_zmat = pd.DataFrame(columns=cols, index=np.arange(0, len(zmat), 1))

    if ENERGY:
        for i in range(len(zmat)):
            new_zmat.iloc[i] = zmat[i].iloc[:, 2].tolist()+zmat[i].iloc[:, 4].tolist()+zmat[i].iloc[:, 6].tolist()+[energies[i]]
        
        set_labels = new_zmat.columns.to_list()[6:-1]
    else:
        for i in range(len(zmat)):
            new_zmat.iloc[i] = zmat[i].iloc[:, 2].tolist()+zmat[i].iloc[:, 4].tolist()+zmat[i].iloc[:, 6].tolist()
        
        set_labels = new_zmat.columns.to_list()[6:]

    new_zmat = deg_to_rad(new_zmat, set_labels)
    new_zmat = rephase(new_zmat)

    new_zmat = new_zmat.drop(["bond0origin", "angle0e_z", "angle2e_z", "dihedral0e_x",
                              "dihedral2e_x", "dihedral3e_x"], axis=1)
    new_zmat = new_zmat.to_numpy(dtype=np.float32)

    return torch.from_numpy(new_zmat).float() 

def get_pos_energy(zmat):
    u_tensor = zmat[:, -1]
    x_tensor = zmat[:, :-1]
    return x_tensor, u_tensor

def shuffle_arr(vs, indexes):
    concat = lambda vs: torch.cat(vs)
    v = concat(vs)
    return v[indexes]

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

class Structure:
    """
    Structure (Object)
    Class storing:
        - the construction table
        - the ase symbols string
        - possibly the calculator? 

    the method to go from internal coordinates to cartesian+molecule:
        - build zmat_matrix and molecule: zmat ---> xyz
          this method uses chemcoord to prepare the data for Ase
    """

    def __init__(self, construction_table=get_construction_table(), 
                symbols=np.full(6, 'Ag')):
        super().__init__()
        
        self.construction_table = construction_table.copy()
        self.symbols = symbols
        self.Natoms = len(self.symbols)                              
        self.calculator = None

      
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
            zmat_values = zmat_values.clone().detach().numpy()
        else:
            zmat_values = zmat_values.copy()
                
        zmat_matrix = self.construction_table.copy()
        
        b = np.zeros(6)
        a = np.zeros(6)
        d = np.zeros(6)
        
        if len(zmat_values)==12:
         
         # reference frame shift - values taken from chemcoord
            b[0] = 1.27
            a[0:2] = np.array([2.21657, 2.21657])
            d[0:3] = np.array([2.21657, 2.21657, 2.21657])

            b[1:] = zmat_values[:5]
            a[2:] = zmat_values[5:9]
            d[3:] = zmat_values[9:]
            
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
            molecule = zmat_matrix.get_cartesian().get_ase_atoms()

        else:
            raise RuntimeError('Data not valid')
      
        return zmat_matrix, molecule 