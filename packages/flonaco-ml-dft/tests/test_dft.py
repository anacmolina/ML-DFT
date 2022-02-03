import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import pandas as pd

from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.adapt import run_mcmc_adapt
from flonacomldft.gaussian_utils import (
    MoG, plot_2d_level
)

if os.path.isdir('/mnt/home/amolina/ceph/'):
    ceph_home = '/mnt/home/amolina/ceph/'
elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'):
    ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'
else:
    raise RuntimeError('Data path not understood')

data_home = ceph_home + 'Database/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype=torch.float32

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)

df = pd.read_csv(data_home + 'is1_zmat.csv')

## code to apply necessary transforms
##  and compute potential energy
## using the one here just for the test
x = df.to_numpy()[:, 1:-1]

args_rnvp = {
    'dim': x.shape[1],
    'n_realnvp_block': 2,
    'block_depth': 1,
    'args_prior': {'type': 'standn'},
    'init_weight_scale': 1e-6,
}

model = RealNVP_MLP(args_rnvp['dim'], 
                    args_rnvp['n_realnvp_block'],
                    args_rnvp['block_depth'],
                    init_weight_scale=args_rnvp['init_weight_scale'],
                    prior_arg=args_rnvp['args_prior'],
                    device=device)

model_init = copy.deepcopy(model)

# _ = run_mcmc_adapt(model, 
#           target, ## to be replaced 
#           n_iter=10,
#           lr=1e-1,
#           bs=100,
#           use_scheduler=False,
#         #   step_schedule=100,
#           args_loss={'type': 'fwd', 'samp': 'direct'},
#           estimate_tau=False,
#           return_all_xs=True,
#           save_splits=10,
#           grad_clip=1e4):

