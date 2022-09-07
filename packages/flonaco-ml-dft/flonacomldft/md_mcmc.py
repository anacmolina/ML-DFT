import pickle
import copy
import torch

import matplotlib.pyplot as plt
import numpy as np

from flonacomldft.dft_utils import (
    run_molecular_dynamics
)

from flonacomldft.data_utils import get_path
from flonacomldft.internal_coordinates import (
    get_internal_coordinates,
    Angles_mapping,
    get_pos_energy,
    shuffle_arr
)

from flonacomldft.dft_utils import Structure
from flonacomldft.train_from_data import train
from flonacomldft.mixture import Mixture
from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.sampling import run_metropolis

from ase.io.trajectory import Trajectory
import gpaw.mpi as mpi


# move to dft_utils.py
def run_md_get_zmat(molecule, iterations, file_name, starting=True):
    traj = run_molecular_dynamics(molecule, iterations, file_name, starting)
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

def run_nf(zmat, model):

    #do clone to copy and not touch the memory space
    x_tensor, u_tensor = get_pos_energy(zmat)

    M = Angles_mapping()
    M.inv_mapping(x_tensor)

    model_init = copy.deepcopy(model) # Do I need this?

    _ = train(model, 
           x_tensor,
           n_iter=500,
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




