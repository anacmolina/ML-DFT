from ase.parallel import parprint as print

print('Starting...\n')
print("Loading the libraries...")

import time
import copy

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.train_from_data import train

from flonacomldft.data_utils import get_path
from flonacomldft.internal_coordinates import Angles_mapping
from flonacomldft.collective_variables import get_CVs

from flonacomldft.visualize import plotting_fes_db

print("Done\n")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))

print('random id!', random_id)
print('Date:', date)

print('Device: %s...\n'%device)
print("Loading the database...")

df = pd.read_csv(get_path() + 'is1_lcao_zmat.csv')

print("Done\n")
print("Setting the input...")

U = df.energies
X = df.drop(['energies'], axis=1)

print("Done\n")
print("Loading the training data...")

n = -1
u = U[:n].to_numpy()
x = X[:n].to_numpy()

x_tensor = torch.from_numpy(x).float()
U_tensor = torch.from_numpy(u).float()

print("Labels", x_tensor.shape[1])
print("Samples", x_tensor.shape[0])

cov = torch.cov(x_tensor.T)
mean = x_tensor.mean(0)

M = Angles_mapping()
M.inv_mapping(x_tensor)

print("Done\n")
print("Normalizing flow training")

args_rnvp = {
    'dim': x.shape[1],
    'n_realnvp_block': 2,
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
           n_iter=50,
           lr=5e-1,
           bs=10,
           use_scheduler=False,
           step_schedule=100,
           args_loss={'type': 'fwd', 'samp': 'direct'},
           estimate_tau=False,
           return_all_xs=True,
           save_splits=10,
           grad_clip=1e4)

print("Done\n")
print("Plotting log_losses")

losses = _['losses']
plt.figure(figsize=(7, 5))
plt.plot(list(range(0, len(losses))), np.abs(np.array(losses)))
plt.yscale('log')
plt.ylabel('Losses')
plt.show()

print("Done\n")
print("Sampling and plotting CVs on FES")

models = _['models']
N_samples = 250

x = models[-1].sample(N_samples)
M.mapping(x)
x = x.clone().data.cpu().numpy()
c_, r_ = get_CVs(x)

ax = plotting_fes_db()
ax.plot(c_, r_, 'mo', label='NF proposals')
ax.legend(loc='lower left', fontsize=20)
plt.show()

print("Done\n")
