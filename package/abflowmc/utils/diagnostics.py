### Import modules
import numpy as np
import torch
from ase.units import kB
import arviz as az
from arviz.utils import get_coords, _var_names

class Target_Log_Prob:
    """Class to compute the target log probability of a model"""
    def __init__(self, energy_type, mode_label=None, mlp_model=None, T=350, kB=kB, folder=None):#kB=8.617333262e-5):
        """Initialize the class
        Args:
            energy_type (str): type of energy to use (dft, mlp, dft+mlp)
            mode_label (int): label of the mode to sample
            T (float): temperature in K
            kB (float): Boltzmann constant in eV/K from ase units module
        """
        self.energy_type = energy_type
        self.mode_label = mode_label
        self.mlp_model = mlp_model
        self.T = T
        self.kB = kB
        self.folder = folder
    
    def __dft_target_log_prob__(self, xs):
        """Compute the target log probability using DFT"""
        from abflowmc.dft_calculator import DFTCalculator
        from abflowmc.internal_coordinates import Coordinates_mapping
        
        coord_mapping = Coordinates_mapping(etype=self.energy_type)
        calculator = DFTCalculator()
        calculator.initialize_calculator(foldername=self.folder)
        
        u = torch.zeros(xs.shape[0])
        for i in range(xs.shape[0]):
            molecule, logdetjac = coord_mapping.build_molecule_from_real_centered(xs[i].reshape(1, -1), int(self.mode_label))
            u_ = calculator.calculate_potential_energy(molecule, filename='ag6_{:d}.out'.format(i))                        
            u[i] = coord_mapping.compute_energy_in_new_frame(u_, logdetjac*(-1))

        return - u / (self.kB * self.T)
    
    def __emt_target_log_prob__(self, xs):
        """Compute the target log probability using EMT"""
        from abflowmc.dft_calculator import EMTCalculator
        from abflowmc.internal_coordinates import Coordinates_mapping
        
        coord_mapping = Coordinates_mapping(etype=self.energy_type)
        calculator = EMTCalculator()
        
        u = torch.zeros(xs.shape[0])

        for i in range(xs.shape[0]):
            molecule, logdetjac = coord_mapping.build_molecule_from_real_centered(xs[i].reshape(1, -1), int(self.mode_label))
            u_ = calculator.calculate_potential_energy(molecule)                        
            u[i] = coord_mapping.compute_energy_in_new_frame(u_, logdetjac*(-1))

        return - u / (self.kB * self.T)
    
    def __mlp_target_log_prob__(self, xs):
        """Compute the target log probability using MLP"""
        if self.mlp_model is None:
            raise ValueError('mlp model not initialized')
        return - self.mlp_model(xs) / (self.kB * self.T)

    def target_log_prob(self, xs):
        """Compute the target log probability"""
        if self.energy_type == 'dft':
            return self.__dft_target_log_prob__(xs)
        elif self.energy_type == 'mlp':
            return self.__mlp_target_log_prob__(xs)
        elif self.energy_type == 'emt':
            return self.__emt_target_log_prob__(xs)
        else:
            raise ValueError('energy_type must be either emt, dft or mlp')
    
def get_participation_ratio(prop, target_log_prob, n_prop):
    """
    Compute the participation ratio of a proposal distribution

    Args:
        prop : Proposal distribution with .sample and .nll methods
        target_log_prob : Log probability of the target distribution.
        n_prop : Number of samples from the proposal distribution.
    
    Returns:
        Participation ratio, 1 if proposal is equal to target and < 1 otherwise

    """
    xs = prop.sample(n_prop)
    log_weight = target_log_prob(xs,).squeeze() + prop.nll(xs)
    log_ratio = torch.logsumexp(2 * log_weight, dim=0) - 2 * torch.logsumexp(log_weight, dim=0) 
    
    return torch.exp(-log_ratio) / n_prop

def get_participation_ratio_from_nlls(us, nlls, kB, T):
    n_prop = us.shape[0]
    log_weight = -us.squeeze()/(kB*T) + nlls
    log_ratio = torch.logsumexp(2 * log_weight, dim=0) - 2 * torch.logsumexp(log_weight, dim=0)
    
    return torch.exp(-log_ratio) / n_prop

def R_hat(chains, burn_in=100, n_split=10, var_names=None, filter_vars=None, labels=None):
    
    idata = az.convert_to_inference_data(chains[:, burn_in:, :])
    coords = {}
    data = get_coords(az.convert_to_dataset(idata, group="posterior"), coords)
    
    var_names = _var_names(var_names, data, filter_vars)
    n_draws = data.dims["draw"]
    n_samples = n_draws * data.dims["chain"]
    first_draw = data.draw.values[0] # int of where where things should start

    ## Compute where to split the data to diagnostic the convergence
    xdata = np.linspace(n_samples / n_split, n_samples, n_split)
    draw_divisions = np.linspace(n_draws // n_split, n_draws, n_split, dtype=int)

    rhat_s = np.stack([np.array(az.rhat(data.sel(draw=slice(first_draw + draw_div)),
                    var_names=var_names,
                    method="rank",
                )['x'])
        for draw_div in draw_divisions
        ])
    
    return rhat_s, draw_divisions, labels



    