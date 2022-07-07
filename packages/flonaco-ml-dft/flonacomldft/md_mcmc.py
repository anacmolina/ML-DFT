
# Get init configs <----------|  
# Run MD                      |
# Train NF                    |
# Run MCMC -------------------|

import pickle

import matplotlib.pyplot as plt
import numpy as np

from flonacomldft.md_utils import (
    get_is1, 
    get_is2, 
    get_path,
    shuffle_arr,
    run_molecular_dynamics, 
    get_internal_coordinates
)

from flonacomldft.dft_utils import Angles_mapping, Structure
from flonacomldft.train_from_data import train

from flonacomldft.training_utils import run_NF
from flonacomldft.mixture import Mixture

from flonacomldft.real_nvp_mlp import RealNVP_MLP
import copy
import torch

from ase.io.trajectory import Trajectory

import gpaw.mpi as mpi

from flonacomldft.sampling import run_metropolis

def get_pos_energy(zmat):
    u_tensor = zmat[:, -1]
    x_tensor = zmat[:, :-1]
    return x_tensor, u_tensor

# move to dft_utils.py
def run_md_get_zmat(molecule, iterations, file_name):
    traj = run_molecular_dynamics(molecule, iterations, file_name)
    zmat = get_internal_coordinates(traj).detach()
    return zmat

def init_model(zmat):
    
    #Set this as a default for all the files
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    x_tensor = zmat[:, :-1]
    
    cov = torch.cov(x_tensor.T)
    cov = torch.eye(12)*cov.mean()
    mean = x_tensor.mean(0)
    
    args_rnvp = {
        'dim': cov.shape[0],
        'n_realnvp_block': 15,
        'block_depth': 1,
        # 'args_prior': {'type': 'standn'}, # standard Gaussian base
        'args_prior': {'type': 'white', 'cov': cov, 'mean': mean}, # Gaussian with non-trival mean and covariance for base
        'init_weight_scale': 1e-6,
        }

    model = RealNVP_MLP(args_rnvp['dim'], 
                    args_rnvp['n_realnvp_block'],
                    args_rnvp['block_depth'],
                    init_weight_scale=args_rnvp['init_weight_scale'],
                    prior_arg=args_rnvp['args_prior'],
                    device=device)

    return model

def get_models(nf_):
    return [nf_[0]['models'][-1], nf_[1]['models'][-1]]

def run_nf(zmat, model):

    #do clone to copy and not touch the memory space
    x_tensor, u_tensor = get_pos_energy(zmat)

    M = Angles_mapping()
    M.inv_mapping(x_tensor)

    model_init = copy.deepcopy(model) # Do I need this?

    _ = train(model, 
           x_tensor,
           n_iter=100,
           lr=5e-2,
           bs=100,
           use_scheduler=False,
           step_schedule=100,
           args_loss={'type': 'fwd', 'samp': 'direct'},
           estimate_tau=False,
           return_all_xs=True,
           save_splits=10,
           grad_clip=1e4)
    
    M.mapping(x_tensor)

    return _

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

def md_mcmc(molecules, states, models, trainig_data, arg_md, arg_mcmc):

    data = []

    for molecule, state in zip(molecules):
        data_ = run_md_get_zmat(get_is1(), md_iters, str(state))
        data.append(data_)

    



    return 0