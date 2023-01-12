import torch
import numpy as np
import pandas as pd
import chemcoord as cc

from flonacomldft.utils.io_utils import get_path
from flonacomldft.utils.silver_isomers_utils import get_construction_table

def add_phase(tensor, phase = 2 * torch.pi):
    return tensor - phase

def get_labels_from_construction_table(construction_table, all_labels=False):

    construction_table = construction_table.copy()
    construction_table['index'] = construction_table.index
    construction_table = construction_table.copy().astype(str)

    bonds = ('b-'+ construction_table['index'] +'-'+ construction_table['b']).tolist()
    angles = ('a-'+ construction_table['index'] +'-'+ construction_table['b'] + '-' + construction_table['a']).tolist()
    dihedrals = ('d-'+ construction_table['index'] +'-'+ construction_table['b'] + '-' + construction_table['a'] + '-' + construction_table['d']).tolist()

    labels = bonds + angles + dihedrals

    if all_labels:
        return labels
    else:
        for label in labels.copy():
            if 'e' in label or 'o' in label:
                labels.remove(label)

        return labels

def from_molecule_to_zmat_tensor(molecule, construction_table, return_logdetjac=True,
                                    return_potential_energy=True, temperature=None):
    
    zmat_matrix = cc.Cartesian.from_ase_atoms(molecule).get_zmat(construction_table.copy())
    zmat_matrix = zmat_matrix.minimize_dihedrals()
    zmat_values = zmat_matrix.loc[:, ['bond', 'angle', 'dihedral']]
    zmat_values.loc[:, ['angle', 'dihedral']] = zmat_values.loc[:, ['angle', 'dihedral']].apply(np.deg2rad)
    zmat_values = zmat_values.to_numpy()[1:, :]
    zmat_flatten = np.concatenate((zmat_values[:, 0], zmat_values[1:, 1], zmat_values[2:, 2]))

    zmat_flatten = torch.tensor(zmat_flatten).detach().requires_grad_().float()

    if return_logdetjac:
        struct = Structure(construction_table)
        logdetjac = torch.tensor([logdetjac_to_xyz(zmat_flatten, struct)[1]]).float()

        if return_potential_energy:
            kb = 8.617333262e-5
            if temperature is None:
                raise RuntimeError('Include temperature value')
            else:    
                potential_energy = molecule.get_potential_energy() - (kb * temperature) * logdetjac

    if return_logdetjac and return_potential_energy:
        return zmat_flatten, logdetjac, potential_energy
    if return_logdetjac==True and return_potential_energy==False:
        return zmat_flatten, logdetjac
    if return_logdetjac==False and return_potential_energy==False:
        return zmat_flatten

def get_internal_coordinates_from_trajectory(trajectory, construction_table, add_logdetjac=True, add_potential_energy=True, temperature=None):

    xs = []
    for configuration in trajectory:
 
        if add_logdetjac and add_potential_energy:
            x, logdetjac, potential_energy = from_molecule_to_zmat_tensor(configuration, 
                                                        construction_table, 
                                                        return_logdetjac=True, 
                                                        return_potential_energy=True,
                                                        temperature=temperature)
        
            x = torch.cat((x, logdetjac, potential_energy), dim=-1)

        if add_logdetjac==True and add_potential_energy==False:
            x, logdetjac = from_molecule_to_zmat_tensor(configuration, 
                                                        construction_table, 
                                                        return_logdetjac=True, 
                                                        return_potential_energy=True)
            x = torch.cat((x, logdetjac), dim=-1)

        if add_logdetjac==False and add_potential_energy==False:
            x = from_molecule_to_zmat_tensor(configuration, 
                                            construction_table, 
                                            return_logdetjac=False)
        
        xs.append(x)
        
    return torch.stack(xs)
 
def logdetjac_to_xyz(zmat, structure): 
    """"
    zmat: array of internal coordinates for a single configuration (12,)
    """
    zmat_matrix = structure.build_zmat_matrix(zmat)
    det = zmat_matrix.get_grad_cartesian(as_function=False)
    det = det.reshape(structure.Natoms * 3, structure.Natoms * 3)
    return np.linalg.slogdet(det)


class Structure:
    """
    Structure (Object)
    Class storing:
        - the construction table
        - the ase symbols string

    the method to go from internal coordinates to cartesian+molecule:
        - build zmat_matrix : zmat_values (internal coord) ---> zmat_matrix (xyz)
        - build_molecule : zmat_values (internal coord) ---> molecule
          this method uses chemcoord to prepare the data for Ase
    """

    def __init__(self, construction_table=get_construction_table(), 
                symbols=np.full(6, 'Ag')):
        super().__init__()
        
        self.construction_table = construction_table.copy()
        self.symbols = symbols
        self.Natoms = len(self.symbols)                              
      
    def build_zmat_matrix(self, zmat_values):  
        """"
        Build the chemcoord zmat matrix adding default frame position/rotation and 
        angles in degrees from the zmat values (only the internal coordinates with 
        angles in radians).
        
        In:
            zmat_values: array with the values of the internal coordinates of one only
            configuration (12 inputs)
        Out:  
            zmat_matrix: chemcoord df-zmat of coordinates in the basis of the IC
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

        else:
            raise RuntimeError('Data not valid')
      
        return zmat_matrix

    def build_molecule(self, zmat_values):
        """"
        Build the ase molecule to feed in the DFT calculator from the zmat values 
        (only the internal coordinates with angles in radians).
        
        In:
            zmat_values: array with the values of the internal coordinates of one only
            configuration (12 inputs)
        Out:  
            molecule: ase object
        """

        zmat_matrix = self.build_zmat_matrix(zmat_values)
        molecule = zmat_matrix.get_cartesian().get_ase_atoms()

        return molecule 

def save_internal_coordinates_to_csv(xs, construction_table,  add_potential_energy=True, add_logdetjac=True, add_isomer=False, filename='traj.csv', path=get_path()):

        labels = get_labels_from_construction_table(construction_table)
        
        if add_logdetjac:
            labels = labels + ['logdetjac']
            
        if add_potential_energy:
            labels = labels + ['potential_energy']

        if add_isomer:
            labels = labels + ['isomer']

        df = pd.DataFrame(xs.detach().numpy())
        df.columns=labels
        df.to_csv(path + '/' + filename, index=False)