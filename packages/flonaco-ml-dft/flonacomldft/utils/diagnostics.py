### Import modules
import numpy as np
import torch
from flonacomldft.sampling import run_metropolis


def get_acceptance_ratio(init, flow_model, n_chains=5, n_steps=10, id_run=None, energy_type='dft', mlp_model=None, T=300, return_ratios=True):
    """
    Compute acceptance ratio for a given flow model

    Args:
        init : Initial configurations
        flow_model : Flow model
        n_chains : Number of chains
        n_steps : Number of steps
        energy_type : Energy type (dft or mlp)
        mlp_model : MLP model (Default value = None)
        T : Temperature (Default value = 300)
        return_ratios: Return acceptance ratio (Default value = True)
    Returns:
        If True, return acceptance ratio, else return the whole simulation (Default value = True)    
    """

    mh_simulation = run_metropolis(
            model=flow_model,
            init=init,
            n_chains=n_chains,
            n_steps=n_steps,
            id_run=id_run, 
            energy_type=energy_type,
            frac_dft=0.2,
            mlp_models=mlp_model,
            T=T,
            return_ratio=True,
            return_proposals=True,
        )

    if return_ratios:
        return mh_simulation['ratios'].mean(dim=1)
    else:
        return mh_simulation
    
class Target_Log_Prob:
    """Class to compute the target log probability of a model"""
    def __init__(self, energy_type, mode_label=None, mlp_model=None, T=300, kb=8.617333262e-5):
        """Initialize the class
        Args:
            energy_type (str): type of energy to use (dft, mlp, dft+mlp)
            mode_label (int): label of the mode to sample
            T (float): temperature in K
            kb (float): Boltzmann constant in eV/K
        """
        self.energy_type = energy_type
        self.mode_label = mode_label
        self.mlp_model = mlp_model
        self.T = T
        self.kb = kb
    
    def __dft_target_log_prob__(self, xs):
        """Compute the target log probability using DFT"""
        from flonacomldft.dft_calculator import DFTCalculator
        from flonacomldft.internal_coordinates import Coordinates_mapping
        
        coord_mapping = Coordinates_mapping()
        calculator = DFTCalculator()
        calculator.initialize_calculator()
        
        u = torch.zeros(xs.shape[0])
        for i in range(xs.shape[0]):
            molecule, logdetjac = coord_mapping.build_molecule_from_real_centered(xs[i].reshape(1, -1), int(self.mode_label))
            u_ = calculator.calculate_potential_energy(molecule, filename='ag6_{:d}.out'.format(i))                        
            u[i] = coord_mapping.compute_energy_in_new_frame(u_, logdetjac*(-1))

        return - u / (self.kb * self.T)
    
    def __mlp_target_log_prob__(self, xs):
        """Compute the target log probability using MLP"""
        if self.mlp_model is None:
            raise ValueError('mlp model not initialized')
        return - self.mlp_model(xs) / (self.kb * self.T)

    def target_log_prob(self, xs):
        """Compute the target log probability"""
        if self.energy_type == 'dft':
            return self.__dft_target_log_prob__(xs)
        elif self.energy_type == 'mlp':
            return self.__mlp_target_log_prob__(xs)
        else:
            raise ValueError('energy_type must be either dft or mlp')
    
#def get_participation_ratio(prop, target_log_prob, n_prop):
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
    log_weight = target_log_prob(xs, ).squeeze() + prop.nll(xs)
    log_ratio = torch.logsumexp(2 * log_weight, dim=0) - 2 * torch.logsumexp(log_weight, dim=0) 
    return torch.exp(-log_ratio) / n_prop

def get_ESS(x):
    """
    Patching to take convention of axis orders,
    and convert from torch to numpy
    x : (n_iter, m_chaines, dim)
    """
    try:
        x = x.detach().numpy()
    except AttributeError:
        x = x

    x = x.swapaxes(0, 1)
    return my_ESS(x)


def my_ESS(x):
    """
    Compute the effective sample size of estimand of interest.
    Vectorised implementation.
    x : m_chains, n_iter, dim
    Computation of effective sampling size from:
    https://github.com/jwalton3141/jwalton3141.github.io
    following definition from:
    ref Gelman, Andrew, J. B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, 
    and Donald B. Rubin. 2013. Bayesian Data Analysis. Third Edition. London: Chapman & Hall / CRC Press.
    """
    if x.shape < (2,):
        raise ValueError(
            'Calculation of effective sample size'
            'requires multiple chains of the same length.')
    try:
        m_chains, n_iter = x.shape
    except ValueError:
        return [my_ESS(y.T) for y in x.T]

    def variogram(t): return (
        (x[:, t:] - x[:, :(n_iter - t)])**2).sum() / (m_chains * (n_iter - t))

    post_var = my_gelman_rubin(x)
    assert post_var > 0

    t = 1
    rho = np.ones(n_iter)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iter):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t - 1:t + 1]) < 0

        t += 1

    return int(m_chains * n_iter / (1 + 2 * rho[1:t].sum()))


def my_gelman_rubin(x):
    """
    Estimate the marginal posterior variance. Vectorised implementation.
    x : m_chaines, n_iter
    """
    m_chains, n_iter = x.shape

    # Calculate between-chain variance
    B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True)) **
         2).sum() / (m_chains * (n_iter - 1))

    # (over) estimate of variance
    s2 = W * (n_iter - 1) / n_iter + B_over_n

    return s2