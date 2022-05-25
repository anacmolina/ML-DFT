from ase.parallel import parprint as print
import numpy as np

import gpaw.mpi as mpi
import torch
import pickle
import pandas as pd

from flonacomldft.dft_utils import (
    Angles_transformation,
    Structure
)

from flonacomldft.sampling import run_metropolis

from flonacomldft.md_utils import (
    get_path,
    load_from_pickle,
    load_csv, 
    shuffle_arr,
    get_is1,
    get_is2,
    get_internal_coordinates,
    run_molecular_dynamics
)

from flonacomldft.mixture import Mixture

from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.train_from_data import train

ceph_home = get_path()

traj_is1 = run_molecular_dynamics(get_is1(), iters=3, name='is1', i=0)
zmat_is1 = get_internal_coordinates(traj_is1)

print(zmat_is1)

u_tensor = zmat_is1[:, -1]
x_tensor = zmat_is1[:, :-1]

print(x_tensor, u_tensor)

cov = torch.cov(x_tensor.T)
mean = x_tensor.mean(0)

x_tensor = Angles_transformation(x_tensor)
x_tensor.inv_transf()


args_rnvp = {
    'dim': x.shape[1],
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

model_init = copy.deepcopy(model)

_ = train(model, 
           x_tensor,
           n_iter=100,
           lr=5e-2,
           bs=10,
           use_scheduler=False,
           step_schedule=100,
           args_loss={'type': 'fwd', 'samp': 'direct'},
           estimate_tau=False,
           return_all_xs=True,
           save_splits=10,
           grad_clip=1e4)


models = _['models'][-1]


