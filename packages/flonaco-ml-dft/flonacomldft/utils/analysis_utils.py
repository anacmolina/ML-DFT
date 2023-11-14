import torch
import numpy as np

class MD_Properties:
    """Class to get the properties of a MD trajectory as a function of time."""
    def __init__(self, dtype=torch.float32, device='cpu'):
        self.trajectory = None
        self.collective_variables = None
        self.temperature = None
        self.total_energy = None
        self.kinetic_energy = None
        self.potential_energy = None
        self.dtype = dtype
        self.device = device        

    def get_total_energy(self, trajectory):
        """Get the total energy of the system as a function of time."""
        total_energy = torch.tensor([molecule.get_total_energy() for molecule in trajectory], dtype=self.dtype)
        return total_energy.to(self.device)
    
    def get_kinetic_energy(self, trajectory):
        """Get the kinetic energy of the system as a function of time."""
        kinetic_energy = torch.tensor([molecule.get_kinetic_energy() for molecule in trajectory], dtype=self.dtype)
        return kinetic_energy.to(self.device)
    
    def get_potential_energy(self, trajectory):
        """Get the potential energy of the system as a function of time."""
        potential_energy = torch.tensor([molecule.get_potential_energy() for molecule in trajectory], dtype=self.dtype)
        return potential_energy.to(self.device)
    
    def get_temperature(self, trajectory):
        """Get the temperature of the system as a function of time."""
        temperature = torch.tensor([molecule.get_temperature() for molecule in trajectory], dtype=self.dtype)
        return temperature.to(self.device)
    
    def get_collective_variables(self, trajectory):
        """Get the collective variables of the system
        Coordination number and radius of gyration"""

        from flonacomldft.collective_variables import get_cvs_from_traj

        return get_cvs_from_traj(trajectory)

    def compute_all_properties(self, trajectory, add_trajectory_to_object=False, add_cvs_to_object=False):
        """Compute all the properties of the system as a function of time and store them in the class object.
        Args:
            trajectory (list): List of ase.Atoms objects.
        """
        if add_trajectory_to_object:
            self.trajectory = trajectory

        self.temperature = self.get_temperature(trajectory)
        self.total_energy = self.get_total_energy(trajectory)
        self.kinetic_energy = self.get_kinetic_energy(trajectory)
        self.potential_energy = self.get_potential_energy(trajectory)

        if add_cvs_to_object:
            self.collective_variables = self.get_collective_variables(trajectory)

#TODO: Add docstring
class NF_Results:

    def __init__(self, flow_dic, dtype=torch.float32, device='cpu'):
        self.flow_dic = flow_dic
        self.losses = None
        self.train_loss = None
        self.test_loss = None        
        self.part_ratio = None
        self.models = None
        self.collective_variables = None

    def get_losses(self, return_loss_dict=False):
        train_loss = self.flow_dic['losses'][0]
        test_loss = self.flow_dic['losses'][1]

        if return_loss_dict:
            losses = {'train': train_loss, 
                       'test': test_loss}
            return losses
        else:       
            return train_loss, test_loss
        
    def get_part_ratio(self):
        return self.flow_dic['part_ratios']
    
    def get_models(self):
        return self.flow_dic['models']
    
    def get_model(self, model_number=-1):
        return self.flow_dic['models'][model_number]

    def generate_samples(self, n_samples, isomer, model_number=-1, **kwargs):

        model = self.get_model(model_number=model_number)

        from flonacomldft.internal_coordinates import Coordinates_mapping
        coord_mapping = Coordinates_mapping()

        self.real_centered_samples = model.sample(n_samples)
        self.molecules_samples = [coord_mapping.build_molecule_from_real_centered(sample.reshape(1, -1), isomer=isomer)[0] 
                                  for sample in self.real_centered_samples]
        
    def compute_potential_energy(self, model=None, n_samples=None, isomer=None, etype='emt', use_save_conformations=True, **kwargs):
            
        if use_save_conformations and self.molecules_samples is not None:
            n_samples = len(self.molecules_samples)
        else:
            self.generate_samples(n_samples, n_samples, isomer, **kwargs)

        etype = etype
        
        calc = None
        
        if etype == 'emt':
        
            from flonacomldft.dft_calculator import EMTCalculator
            calc = EMTCalculator()
        
        elif etype == 'dft':
        
            from flonacomldft.dft_calculator import DFTCalculator
        
            calc = DFTCalculator()
            calc.initialize_calculator()
        
        else:
        
            raise Warning('Energy type not recognized. Please choose between "emt" or "dft".')  
          
        calc = calc

        us_molecules = torch.zeros((n_samples, 1))
        
        for i in range(n_samples):
            us_molecules[i] = calc.calculate_potential_energy(self.molecules_samples[i])
        
        self.energy_molecules = us_molecules

    def get_collective_variables(self, model=None, n_samples=None, isomer=None, **kwargs):

        from flonacomldft.collective_variables import get_cvs_from_real_centered

        if self.real_centered_samples is not None:
            self.collective_variables = get_cvs_from_real_centered(self.real_centered_samples, isomer=isomer)
        else:
            raise Warning('No samples generated. Please generate samples first.')

        
    def compute_all_results(self, add_samples=True, add_energy=True, n_samples=10, isomer=None, **kwargs):
        
        self.losses = self.get_losses(True)
        self.train_loss = self.losses['train']
        self.test_loss = self.losses['test']
        self.part_ratio = self.get_part_ratio()
        self.models = self.get_models()

        if add_samples:
            self.generate_samples(n_samples=n_samples, isomer=isomer, **kwargs)
            self.get_collective_variables(isomer=isomer, **kwargs)
        
        if add_energy:
            self.compute_potential_energy(**kwargs)


