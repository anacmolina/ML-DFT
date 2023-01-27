import torch
import numpy as np
import pandas as pd
import chemcoord as cc

from flonacomldft.utils.io_utils import get_path
from flonacomldft.utils.silver_isomers_utils import get_construction_table, get_molecule_isomer_minima 

# add a phase for some internal coordinates angles
def add_phase(tensor, phase = 2 * torch.pi):
    return tensor - phase

# map from radians to reals with tan function
class Angles_mapping():
    """
    Class to get the forward and backward mapping of the angles.
    It stores the index at which the angles start in the internal coordinates

    """
    def __init__(self, idx_first_angle=5):
        self.idx_first_angle = idx_first_angle

    def rads_to_reals(self, x_rads, log_det_jac=None):
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

# class for mapping in all frame
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
                symbols=np.full(6, 'Ag')):
        super().__init__()
        
        self.construction_table = construction_table.copy()
        self.symbols = symbols
        self.Natoms = len(self.symbols) 
        self.angles_mappings = Angles_mapping()    
        self.zmat_minima = {
            0: self.get_internal_from_molecule(get_molecule_isomer_minima('ag6_planar'))[0],
            1: self.get_internal_from_molecule(get_molecule_isomer_minima('ag6_3d'))[0]
        } 
        self.kb = 8.617333262e-5 # eV/K

    # TODO: Add description
    def _get_xyz_from_molecule(self, molecule): 
        return cc.Cartesian.from_ase_atoms(molecule)
    
    # TODO: Add description
    def _get_zmat_matrix_from_xyz(self, xyz): 
        return xyz.get_zmat(self.construction_table.copy())
    
    # TODO: Add description
    def _get_zmat_from_zmat_matrix(self, zmat_matrix):

        zmat_matrix = zmat_matrix.minimize_dihedrals()
        zmat = zmat_matrix.loc[:, ['bond', 'angle', 'dihedral']]
        zmat.loc[:, ['angle', 'dihedral']] = zmat.loc[:, ['angle', 'dihedral']].apply(np.deg2rad)
        zmat = zmat.to_numpy()[1:, :]
        zmat = np.concatenate((zmat[:, 0], zmat[1:, 1], zmat[2:, 2]))

        return torch.tensor(zmat).float()
    
    # TODO: Is this function important?
    def get_zmat_from_molecule(self, molecule):
        xyz = self._get_xyz_from_molecule(molecule)
        zmat = self._get_zmat_from_zmat_matrix(
            self._get_zmat_matrix_from_xyz(xyz)
        )
        return zmat
      
    def _build_zmat_matrix_from_zmat(self, zmat): #NOTE: It's working
        # TODO: Rebuild this using read_xyz from chemcoord  
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

        if torch.is_tensor(zmat):
            zmat = zmat.clone().detach().numpy()
        else:
            zmat = zmat.copy()
                
        zmat_matrix = self.construction_table.copy()
        
        b = np.zeros(6)
        a = np.zeros(6)
        d = np.zeros(6)
        
        if len(zmat)==12:
         
            # reference frame shift - values taken from chemcoord
            b[0] = 1.27
            a[0:2] = np.array([2.21657, 2.21657])
            d[0:3] = np.array([2.21657, 2.21657, 2.21657])

            b[1:] = zmat[:5]
            a[2:] = zmat[5:9]
            d[3:] = zmat[9:]
            
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

    def _build_xyz_from_zmat_matrix(self, zmat_matrix):
        return zmat_matrix.get_cartesian()

    def _build_molecule_from_xyz(self, xyz):
        return xyz.get_ase_atoms()

    #TODO: It's this function important?
    def build_molecule_from_zmat(self, zmat):
        """"
        Build the ase molecule to feed in the DFT calculator from the zmat values 
        (only the internal coordinates with angles in radians).
        
        Args:
            zmat_values: array with the values of the internal coordinates of one only
            configuration (12 inputs)
        
        Returns:  
            molecule: ase object
        """

        xyz = self._build_xyz_from_zmat_matrix(
            self._build_zmat_matrix_from_zmat(zmat)
        )
        
        return xyz.get_ase_atoms()

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

        return xyz_rebuilt

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
        xyz = self._orient_and_center_xyz(xyz)
        
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
            ).float()

        if requires_grad:
            zmat.requires_grad_()

        if logdetjac is None:
            logdetjac = 0

        logdetjac_to_xyz = torch.tensor([self.logdetjac_zmat_to_xyz(zmat)[1]]).float()
        logdetjac += logdetjac_to_xyz
        
        return zmat, logdetjac
        
    def get_cartesian_from_internal(self, zmat, logdetjac=None):
        
        xyz = self._build_xyz_from_zmat_matrix(
            self._build_zmat_matrix_from_zmat(zmat)
            )

        if logdetjac is None:
            logdetjac = 0
        
        logdetjac_to_zmat = torch.tensor([self.logdetjac_xyz_to_zmat(xyz)[1]]).float()
        logdetjac += logdetjac_to_zmat
        
        return xyz, logdetjac

    def compute_energy_in_new_frame(self, energy, logdetjac, temperature=300):
        return energy - (self.kb * temperature) * logdetjac


    def get_internal_from_molecule(self, molecule, return_potential_energy=False,
                                temperature=300, requires_grad=False):
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
            

    def get_internal_from_trajectory(self, trajectory, add_logdetjac=True, 
                                     add_potential_energy=True, temperature=None,
                                     max_samples=None):
        """
        Loops over the previous function to compute the internal coordinates for a whole trajectory.
        """

        xs = []
        for m,molecule_configuration in enumerate(trajectory):
    
            if add_logdetjac and add_potential_energy:
                x, logdetjac, potential_energy = self.get_internal_from_molecule(molecule_configuration, 
                                                            return_logdetjac=True, 
                                                            return_potential_energy=True,
                                                            temperature=temperature)
            
                x = torch.cat((x, logdetjac, potential_energy), dim=-1)

            if add_logdetjac==True and add_potential_energy==False:
                x, logdetjac = self.get_internal_from_molecule(molecule_configuration, 
                                                            return_logdetjac=True, 
                                                            return_potential_energy=False)
                x = torch.cat((x, logdetjac), dim=-1)

            if add_logdetjac==False and add_potential_energy==False:
                x = self.get_internal_from_molecule(molecule_configuration, 
                                                      return_logdetjac=False)
            
            xs.append(x)
            
            if max_samples is not None:
                if m >= max_samples:
                    break

        return torch.stack(xs)
    
    def get_real_centered_from_internal(self, zmats, logdetjacs, isomer, 
                                        temperature=300, energies=None):
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
        zmats = zmats - self.zmat_minima[isomer]
        zmat_reals, logdetjacs_angle = self.angles_mappings.rads_to_reals(zmats)
        logdetjacs += logdetjacs_angle

        if energies is not None:
            energies = energies - (self.kb * temperature) * logdetjacs_angle
            return zmat_reals, logdetjacs, energies

        return zmat_reals, logdetjacs

    def get_internal_from_real_centered(self, reals, logdetjacs, isomer, 
                                        temperature=300, energies=None):
        
        zmats, logdetjacs_angle = self.angles_mappings.reals_to_rads(reals)
        logdetjacs += logdetjacs_angle

        zmats = reals + self.zmat_minima[isomer]

        if energies is not None:
            energies = energies - (self.kb * temperature) * logdetjacs_angle
            return zmats, logdetjacs, energies

        return zmats, logdetjacs
        

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


    #def get_internal_from_cartesian(self, xyz, return_logdetjac=True):
#
    #    # zmat_matrix = xyz.get_zmat(self.construction_table.copy())
    #    # zmat_matrix = zmat_matrix.minimize_dihedrals()
    #    # zmat_values = zmat_matrix.loc[:, ['bond', 'angle', 'dihedral']]
    #    # zmat_values.loc[:, ['angle', 'dihedral']] = zmat_values.loc[:, ['angle', 'dihedral']].apply(np.deg2rad)
    #    # zmat_values = zmat_values.to_numpy()[1:, :]
    #    # zmat_flatten = np.concatenate((zmat_values[:, 0], zmat_values[1:, 1], zmat_values[2:, 2]))
#
    #    if return_logdetjac:
    #        logdetjac = torch.tensor([self.logdetjac_internal_to_xyz(zmat_flatten)[1]]).float()
    #    
    #    return torch.tensor(zmat_flatten).float()


#    def get_internal_from_molecule(self, molecule, return_logdetjac=False,
#                                   return_potential_energy=False, temperature=None,
#                                   requires_grad=False):
#        """"
#        Computes the internal coordinate tensor from a molecule and a construction table.
#        If return_logdetjac is True, the logdetjac is also computed.
#        If return_potential_energy is True, the potential energy is also computed.
#
#        Args:
#            molecule (ase.Atoms): molecule single configuration in ASE format
#        
#        Returns:
#            zmat (torch.tensor): flattened zmat tensor
#            (opt) logdetjac (torch.tensor): logdetjac tensor
#            (opt) potential_energy (array): potential energy tensor
#        """
#        
#        xyz = cc.Cartesian.from_ase_atoms(molecule)
#        zmat = self.get_internal_from_cartesian(xyz)
#    
#        if return_logdetjac:
#            logdetjac = torch.tensor([self.logdetjac_internal_to_xyz(zmat)[1]]).float()
#
#            if return_potential_energy:
#                if temperature is None:
#                    raise RuntimeError('Include temperature value and ask for logdetjac computation')
#                else:    
#                    potential_energy = molecule.get_potential_energy() - (self.kb * temperature) * logdetjac
#        
#        if requires_grad:
#            zmat.requires_grad_()
#
#        if return_logdetjac and return_potential_energy:
#            return zmat, logdetjac, potential_energy
#        if return_logdetjac and not return_potential_energy:
#            return zmat, logdetjac
#        else:
#            return zmat


    #def get_molecule_from_internal(self, zmat):
    #    
#
    #    zmat_matrix = self._build_zmat_matrix_from_zmat(zmat)
    #    molecule = zmat_matrix.get_cartesian().get_ase_atoms()
#
    #    return molecule 