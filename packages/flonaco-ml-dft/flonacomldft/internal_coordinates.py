# libraries

import os
import torch
import numpy as np
import pandas as pd
import chemcoord as cc
from ase.units import kB

from flonacomldft.utils.silver_isomers_utils import get_construction_table, get_molecule_isomer_minima
from flonacomldft.collective_variables import get_collective_variables

def add_phase(tensor, phase = 2 * torch.pi):
    """Shift the angles by phase to avoid discontinuities
    Args:
        tensor (torch.tensor): tensor of angles
        phase (float): phase to add to the angles
    Returns:
        tensor (torch.tensor): tensor of angles with phase added"""
    return tensor - phase


class Angles_mapping():
    """
    Class to get the forward and backward mapping of the angles: 
    it maps from radians to reals with tan function.
    It stores the index at which the angles start in the internal coordinates

    """
    def __init__(self, idx_first_angle=5):
        """
        Args:
            idx_first_angle (int): index at which the angles start in the internal coordinates
        """
        self.idx_first_angle = idx_first_angle

    def rads_to_reals(self, x_rads, log_det_jac=None):
        """
        Transform from radians to reals using the tan function
        Args:
            x_rads (torch.tensor): tensor of angles in radians
            log_det_jac : log determinant of the jacobian
        Returns:
            x_reals (torch.tensor): tensor of angles in reals
        """
        if log_det_jac is None:
            log_det_jac = 0

        x_reals = x_rads.clone()
        x_reals[:, self.idx_first_angle:] = x_rads[:, self.idx_first_angle:].tan()
        log_det_jac += torch.log(1 + x_reals[:, self.idx_first_angle:]**2).sum(-1)
        return x_reals, log_det_jac
        
    def reals_to_rads(self, x_reals, log_det_jac=None):
        if log_det_jac is None:
            log_det_jac = 0

        x_rads = x_reals.clone()
        x_rads[:, self.idx_first_angle:] = x_reals[:, self.idx_first_angle:].arctan()
        log_det_jac -= torch.log(1 + x_reals[:, self.idx_first_angle:]**2).sum(-1)
        return x_rads, log_det_jac

### Class for mapping in all frame
class Coordinates_mapping():
    """
    Coordinates_mapping (Object)
    Class storing:
        - the construction table
        - the ase symbols string

    the method to go from internal coordinates to cartesian+molecule:
        - _build zmat_matrix : zmat_values (internal coord) ---> zmat_matrix (xyz)
        - get_molecule_from_internal : zmat_values (internal coord) ---> molecule (xyz)
        - get_internal_from_molecule : molecule (xyz) ---> zmat_values (internal coord)
    """

    def __init__(self, construction_table=get_construction_table(), 
                symbols=np.full(6, 'Ag'), etype='dft'):
        super().__init__()
        
        self.construction_table = construction_table.copy()
        self.symbols = symbols
        self.Natoms = len(self.symbols) 
        self.angles_mappings = Angles_mapping()
        if 'emt' in etype:
            minima_name = 'emt_'
        else:
            minima_name = ''    
        self.zmat_minima = {
            0: self.get_internal_from_molecule(get_molecule_isomer_minima(minima_name+'is0'))[0],
            1: self.get_internal_from_molecule(get_molecule_isomer_minima(minima_name+'is1'))[0] #TODO: becareful with the silver minima file
        } 
        
        self.kB = kB # eV/K

    def _get_xyz_from_molecule(self, molecule):
        """"
        Get the Cartesian XYZ object from a ASE Atoms object (molecule)
        
        Args:
            molecule: ASE Atoms object
        Returns:  
            xyz: chemcoord Cartesian XYZ object
        """
        return cc.Cartesian.from_ase_atoms(molecule)
    
    def _get_zmat_matrix_from_xyz(self, xyz):
        """"
        Get the ZMAT matrix object from a Cartesian XYZ object
        
        Args:
            xyz: ASE Atoms object Cartesian XYZ object
        Returns:  
            zmat_matrix: chemcoord ZMAT matrix object
        """ 
        return xyz.get_zmat(self.construction_table.copy())
    
    def _get_zmat_from_zmat_matrix(self, zmat_matrix):
        """"
        Returns only the bonds, angles and dihedrals of the ZMAT matrix
        and turns it into a zmat tensor
        
        Args:
            ZMAT matrix: chemcoord ZMAT matrix object
        Returns:  
            zmat (torch.tensor - (dims)): zmat tensor
        """

        zmat_matrix = zmat_matrix.minimize_dihedrals()
        zmat = zmat_matrix.loc[:, ['bond', 'angle', 'dihedral']]
        zmat.loc[:, ['angle', 'dihedral']] = zmat.loc[:, ['angle', 'dihedral']].apply(np.deg2rad)
        zmat = zmat.to_numpy()[1:, :]
        
        zmat = torch.tensor(
            np.concatenate((zmat[:, 0], zmat[1:, 1], zmat[2:, 2])) 
            ).requires_grad_()

        return zmat.to(torch.float32)
    
    # NOTE: This function it's not being used
    def get_zmat_from_molecule(self, molecule):
        """"
        Get the Cartesian XYZ object from a ASE Atoms object (molecule)
        
        Args:
            molecule: ASE Atoms object
        Returns:  
            zmat (torch.tensor - (dims)): zmat tensor
        """
        xyz = self._get_xyz_from_molecule(molecule)
        zmat = self._get_zmat_from_zmat_matrix(
            self._get_zmat_matrix_from_xyz(xyz)
        )
        return zmat
      
    def _build_zmat_matrix_from_zmat(self, zmat): #NOTE: It's working
        """"
        Build the chemcoord zmat matrix adding default frame position/rotation and 
        angles in degrees from the zmat values (only the internal coordinates with 
        angles in radians).
        
        Args:
            zmat_values: array with the values of the internal coordinates of one only
            configuration (12 inputs)
        Returns:  
            zmat_matrix: chemcoord df-zmat of coordinates in the basis of the IC
        """
        
        if zmat.shape[0] == 3*self.Natoms - 6:

            zmat = zmat.clone().to(torch.double)
            zmat_matrix = self.construction_table.copy()

            zmat_matrix.insert(0, "atom", self.symbols.copy(), True)

            zmat_ref = torch.tensor([1.27, 2.21657, 2.21657]).to(torch.double)

            b = torch.cat((zmat_ref[0].unsqueeze(0), zmat[:self.Natoms-1])).detach().numpy()
            a = torch.cat((zmat_ref[1].repeat(2), zmat[self.Natoms-1:2*self.Natoms-3])).detach().numpy()
            d = torch.cat((zmat_ref[1].repeat(3),zmat[2*self.Natoms-3:])).detach().numpy()

            a = np.rad2deg(a)
            d = np.rad2deg(d)

            zmat_matrix.insert(2, "bond", b, True)
            zmat_matrix.insert(4, "angle", a, True)
            zmat_matrix.insert(6, "dihedral", d, True)

            zmat_matrix = cc.Zmat(zmat_matrix)

        else:
            raise RuntimeError('Data not valid')
      
        return zmat_matrix

    def _build_xyz_from_zmat_matrix(self, zmat_matrix):
        """"
        Build the Cartesian XYZ object from a ZMAT matrix object
        
        Args:
            zmat_matrix: chemcoord ZMAT matrix object
        Returns:  
            xyz: Cartesian XYZ object

        """ 
        return zmat_matrix.get_cartesian().sort_index()

    def _build_molecule_from_xyz(self, xyz):
        """"
        Build the ASE Atoms object from a Cartesian XYZ object
        
        Args:
            xyz: Cartesian XYZ object
        Returns:  
            molecule: ASE Atoms object

        """ 
        return xyz.get_ase_atoms()

    def build_molecule_from_zmat(self, zmat):
        """"
        Build the ase molecule to feed in the DFT calculator from the zmat values 
        (only the internal coordinates with angles in radians).
        
        Args:
            zmat_values: array with the values of the internal coordinates of one only
            configuration (12 inputs)
        
        Returns:  
            molecule: ASE Atoms object
        """

        xyz = self._build_xyz_from_zmat_matrix(
            self._build_zmat_matrix_from_zmat(zmat)
        )
        
        return xyz.get_ase_atoms()

    def logdetjac_zmat_to_xyz(self, zmat): 
        """"
        Args:
            zmat_values: array of internal coordinates for a single configuration (12,)

        Returns:
            s, logdetjac: sign and absolute value of log of the determinant of the jacobian of 
                          the transformation from internal to cartesian coordinates
        """
        zmat_matrix = self._build_zmat_matrix_from_zmat(zmat)
        det = zmat_matrix.get_grad_cartesian(as_function=False)
        det = det.reshape(self.Natoms * 3, self.Natoms * 3)
        
        return np.linalg.slogdet(det)

    def logdetjac_xyz_to_zmat(self, xyz):
            
        """"
        Args:
            xyz: chemcoord Cartesian object

        Returns:
            s, logdetjac: sign and absolute value of log of the determinant of the jacobian of 
                          the transformation from internal to cartesian coordinates
        """
        xyz = self._orient_and_center_xyz(xyz.copy())
        xyz = xyz.loc[self.construction_table.index]
        det = xyz.get_grad_zmat(self.construction_table.copy(), as_function=False)
        det = det.reshape(self.Natoms * 3, self.Natoms * 3)
        
        return np.linalg.slogdet(det)

    def get_internal_from_cartesian(self, xyz, logdetjac=None,
                                   requires_grad=False): 
        """
        Args:
            xyz: chemcoord Cartesian object
        Returns:
            zmat: tensor
        """

        zmat_matrix = self._get_zmat_matrix_from_xyz(xyz)
        zmat = self._get_zmat_from_zmat_matrix(
            zmat_matrix
            )

        if requires_grad:
            zmat.requires_grad_()

        if logdetjac is None:
            logdetjac = 0

        logdetjac_to_xyz = torch.tensor([self.logdetjac_zmat_to_xyz(zmat)[1]])
        logdetjac += logdetjac_to_xyz
        
        return zmat, logdetjac
        
    def get_cartesian_from_internal(self, zmat, logdetjac=None):
        
        xyz = self._build_xyz_from_zmat_matrix(
            self._build_zmat_matrix_from_zmat(zmat)
            )

        if logdetjac is None:
            logdetjac = 0
        
        logdetjac_to_zmat = torch.tensor([self.logdetjac_xyz_to_zmat(xyz)[1]])
        logdetjac += logdetjac_to_zmat
        
        return xyz, logdetjac

    def compute_energy_in_new_frame(self, energy, logdetjac, temperature=350):
        return energy - (self.kB * temperature) * logdetjac

    def get_internal_from_molecule(self, molecule, return_potential_energy=False,
                                temperature=350, requires_grad=False):
        """"
        Computes the internal coordinate tensor from a molecule and a construction table.
        If return_logdetjac is True, the logdetjac is also computed.
        If return_potential_energy is True, the potential energy is also computed.

        Args:
            molecule (ase.Atoms): molecule single configuration in ASE format
        
        Returns:
            zmat (torch.tensor): flattened zmat tensor
            (opt) logdetjac (torch.tensor): logdetjac tensor
            (opt) potential_energy (array): potential energy tensor
        """
        
        xyz = self._get_xyz_from_molecule(molecule)
        zmat, logdetjac_to_xyz = self.get_internal_from_cartesian(xyz)
    
        
        if return_potential_energy:
            if temperature is None:
                raise RuntimeError('Include temperature value and ask for logdetjac computation')
            else:    
                potential_energy = self.compute_energy_in_new_frame(molecule.get_potential_energy(),
                                                        logdetjac_to_xyz, 
                                                        temperature)
        
        if requires_grad:
            zmat.requires_grad_()

        if return_potential_energy:
            return zmat, logdetjac_to_xyz, potential_energy
        else:
            return zmat, logdetjac_to_xyz
            

    def get_internal_from_trajectory(self, trajectory, isomer=None,
                                     add_potential_energy=True, 
                                     add_cvs=True, temperature=None,
                                     max_samples=None):
        """
        Loops over the previous function to compute the internal coordinates for a whole trajectory.
        """

        zmats = []
        for m, molecule_configuration in enumerate(trajectory):
    
            if add_potential_energy:
                zmat, logdetjac, potential_energy = self.get_internal_from_molecule(molecule_configuration, 
                                                            return_potential_energy=True,
                                                            temperature=temperature)
                if isomer is not None:
                    isomer = torch.tensor([isomer])
                    zmat = torch.cat((zmat, potential_energy, isomer), dim=-1 )
                else:   
                    zmat = torch.cat((zmat, potential_energy), dim=-1 )
                
                zmat = torch.cat((zmat, logdetjac), dim=-1)

            else:
                zmat, logdetjac = self.get_internal_from_molecule(molecule_configuration, 
                                                            return_potential_energy=False)
                if isomer is not None:
                    isomer = torch.tensor([isomer])
                    zmat = torch.cat((zmat, isomer), dim=-1 )

                zmat = torch.cat((zmat, logdetjac), dim=-1)

            if add_cvs:
                cv = self.get_collective_variables_from_molecule(molecule_configuration)
                zmat = torch.cat((zmat, cv), dim=-1)


            zmats.append(zmat)
            
            if max_samples is not None:
                if m >= max_samples:
                    break

        return torch.stack(zmats)

    def _orient_and_center_xyz(self, xyz):
        """
        Args:
            xyz: chemcoord Cartesian object
        Returns:
            xyz: chemcoord Cartesian object
        """

        zmat = self._get_zmat_from_zmat_matrix(
            self._get_zmat_matrix_from_xyz(xyz)
        )
        xyz_rebuilt = self._build_zmat_matrix_from_zmat(zmat).get_cartesian()

        return xyz_rebuilt.sort_index()

    def reorient_and_center_molecule(self, molecule): #NOTE: It's working
        """
        Args:
            molecule: ase atoms (molecule) object
        Returns:
            xyz: chemcoord Cartesian object
        """

        xyz = self._get_xyz_from_molecule(molecule)
        xyz_rebuilt = self._orient_and_center_xyz(xyz)

        return xyz_rebuilt.get_ase_atoms()
    
    def get_real_centered_from_internal(self, zmats, logdetjacs=None, isomer=None, 
                                        temperature=350, energies=None):
        """    
        Centers the internal coordinates and takes in real space from raw zmat

        Args:
            zmats (torch.tensor - (nsamples, dims)): zmat tensor
            logdetjacs (torch.tensor) - (nsamples): logdetjac tensor from xyz -> zmat
            isomer (str): isomer name '0' or '1'
            temperature (float): temperature value
            energies (torch.tensor - (nsamples)): potential energy tensor
        
        Returns:
            zmat_reals (torch.tensor - (nsamples, dims)): zmat tensor in real space
            logdetjacs (torch.tensor - (nsamples)): logdetjac tensor from xyz -> zmat in real space (angles mapped)
        """
        if logdetjacs is None:
            logdetjacs = 0

        zmats = zmats - self.zmat_minima[isomer]
        reals, logdetjacs_angle = self.angles_mappings.rads_to_reals(zmats)
        logdetjacs += logdetjacs_angle

        if energies is not None:
            energies = energies - (self.kB * temperature) * logdetjacs_angle
            return reals, logdetjacs, energies

        return reals, logdetjacs

    def get_internal_from_real_centered(self, reals, logdetjacs=None, isomer=None, 
                                        temperature=350, energies=None):
        """
        Get the internal coordinates from the real centered coordinates
        Args:
            reals (torch.tensor): real centered coordinates
            logdetjacs (torch.tensor): logdetjac tensor
            isomer (int): isomer label
            temperature (float): temperature value
            energies (torch.tensor): potential energy tensor
        Returns:
            zmats (torch.tensor): internal coordinates
            logdetjacs (torch.tensor): logdetjac tensor
        """
        
        if logdetjacs is None:
            logdetjacs = 0

        reals, logdetjacs_angle = self.angles_mappings.reals_to_rads(reals)
        logdetjacs = logdetjacs + logdetjacs_angle.requires_grad_(True)

        zmats = reals + self.zmat_minima[isomer]

        if energies is not None:
            energies = energies - (self.kB * temperature) * logdetjacs_angle
            return zmats, logdetjacs, energies

        return zmats, logdetjacs
    
    def build_molecule_from_real_centered(self, x, isomer, temperature=350):
        """
        Builds the molecule from the real centered coordinates
        Args:
            x (torch.tensor): real centered coordinates
            isomer (int): isomer label
            temperature (float): temperature value
        Returns:
            molecule (ase.Atoms): molecule object
        """

        zmat, logdetjac = self.get_internal_from_real_centered(x, isomer=isomer, temperature=temperature)
        xyz, logdetjac = self.get_cartesian_from_internal(zmat[0], logdetjac)
        molecule = self._build_molecule_from_xyz(xyz)

        return molecule, logdetjac

    def get_collective_variables_from_molecule(self, molecule):
        """
        Get the collective variables from a molecule
        Args:
            molecule: ASE Atoms object
        Returns:
            cv (torch.tensor): collective variables tensor
        """
            
        return torch.tensor(get_collective_variables(molecule)).squeeze()

    def get_collective_variables_from_trajectory(self, trajectory, max_samples=None):
        """
        Get the collective variables from a trajectory
        Args:
            trajectory: ASE trajectory object
            max_samples: maximum number of samples to consider
        Returns:
            cvs (torch.tensor): collective variables tensor
        """
        
        cvs = []
        
        for m, molecule_configuration in enumerate(trajectory):
            cv = get_collective_variables(molecule_configuration)
            cvs.append(torch.tensor(cv))
            
            if max_samples is not None:
                if m >= max_samples:
                    break

        return torch.stack(cvs)

    def get_collective_variables_from_zmat(self, x):
        """
        Get the collective variables from the internal coordinates
        Args:
            x: internal coordinates
        Returns:
            cv (torch.tensor): collective variables tensor
        """

        if len(x.shape) > 1:
            cvs = []
            for xi in x:
                molecule = self.build_molecule_from_zmat(xi)
                cv = get_collective_variables(molecule)
                cvs.append(torch.tensor(cv))

            return torch.stack(cvs).squeeze()
                
        else:
            molecule = self.build_molecule_from_zmat(x)
            return torch.tensor(get_collective_variables(molecule)).squeeze()

    def get_collective_variables_from_real_centered(self, x, isomer):
        """
        Get the collective variables from the real centered coordinates
        Args:
            x: real centered coordinates
            isomer: isomer label
        Returns:
            cv (torch.tensor): collective variables tensor
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        zmat, logdetjac = self.get_internal_from_real_centered(x, isomer=isomer)     
                
        return self.get_collective_variables_from_zmat(zmat)
    

        
def get_labels_from_construction_table(construction_table, all_labels=False):
    """
    Get the labels from the construction table
    Args:
        construction_table: construction table
        all_labels: boolean to get all labels
    Returns:
        labels: list of labels
    """

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

def save_internal_coordinates_to_csv(xs, 
                                     columns=None, 
                                     construction_table=None,  
                                     add_potential_energy=True, 
                                     add_logdetjac=True, 
                                     add_isomer=True, 
                                     add_cvs=True, 
                                     filename='traj.csv', 
                                     path=os.getcwd()):
    
    """
    Save the internal coordinates to a csv file
    Args:
        xs: internal coordinates
        columns: list of columns
        construction_table: construction table
        add_potential_energy: boolean to add potential energy
        add_logdetjac: boolean to add logdetjac
        add_isomer: boolean to add isomer
        add_cvs: boolean to add collective variables
        filename: name of the file
        path: path to save the file
    Returns:
        None    
    """

    if (columns is None) and (construction_table is not None):
        
        columns = get_labels_from_construction_table(construction_table)
    
    elif (columns is None) and (construction_table is None):
        
        raise RuntimeError("Can not define columns for dataframe")        

    if add_potential_energy:
        columns = columns + ['potential_energy']

    if add_isomer:
        columns = columns + ['isomer']

    if add_logdetjac:
        columns = columns + ['logdetjac']

    if add_cvs:
        columns = columns + ['C', 'R']

    df = pd.DataFrame(xs.detach().numpy())
    df.columns=columns
    df.to_csv(path + '/' + filename, index=False)

#TODO: Check if this function is being used
def join_data(xs, energies, isomers, logdetjacs=None):

    data = torch.cat((xs, 
            energies.reshape(-1, 1), 
            isomers.reshape(-1, 1)), 
        dim=1).to(torch.float32)
    
    if logdetjacs is not None:
        data = torch.cat((data,
            logdetjacs.reshape(-1, 1)), dim=1)

    return data

def get_collective_variables_from_xs(xss, isomerss):
    """
    Get the collective variables from real centered coordinates
    Args:
        xss: real centered coordinates
        isomerss: isomer labels
    Returns:
        cvss: collective variables tensor
    """
    cvss = []

    coord_mapping = Coordinates_mapping()

    for xs, isomers in zip(xss, isomerss):
        cvs = []
        for x, isomer in zip(xs, isomers):
            molecule = coord_mapping.build_molecule_from_real_centered(x.reshape(1, -1), 
                                                                       isomer=isomer.int().item())[0]
            cvs.append(torch.tensor(get_collective_variables(molecule)))
        cvss.append(torch.stack(cvs))
    
    return torch.stack(cvss)


def load_DFTAdaptive_folder(folder_path, 
                            n_runs, 
                            n_steps, 
                            n_chains, 
                            isomer, 
                            temperature=350):
    
    """
    Load atoms objects from a folder with the DFTAdaptive output files and change to internal coordinates
    Args:
        folder_path: path to the folder
        n_runs: number of runs
        n_steps: number of steps
        n_chains: number of chains
        isomer: isomer label
        temperature: temperature value
    Returns:
        zmats: internal coordinates
    """

    zmats = []
    coord_mapping = Coordinates_mapping()
    from ase.io import read
    
    for i in range(n_runs):
    
        for j in range(n_steps):
    
            for k in range(n_chains):

                try:
    
                    file = 'ag6_{:d}_{:d}_{:d}.out'.format(i, j, k)
    
                    molecule = read(folder_path+'/'+file)
                    u = molecule.get_potential_energy()
                    cv = torch.from_numpy(get_collective_variables(molecule))
    
                    zmat, logdetjac = coord_mapping.get_internal_from_molecule(molecule, temperature=temperature)
                    u_zmat = coord_mapping.compute_energy_in_new_frame(u, logdetjac=logdetjac, temperature=temperature)
    
                    row = torch.cat((zmat.reshape(1, -1), 
                                        u_zmat.reshape(1, -1), 
                                        torch.tensor([isomer]).reshape(1, -1), 
                                        logdetjac.reshape(1, -1), 
                                        cv.reshape(1, -1)), dim=1
    
                        )
    
                    zmats.append(row)

                except:

                    continue
    
    return torch.stack(zmats).squeeze()