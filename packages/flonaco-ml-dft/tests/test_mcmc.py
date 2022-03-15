from ase.parallel import parprint as print

print('Starting...\n')

print("Loading the libraries...")
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import sys
import pandas as pd
import pickle

from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.train_from_data import train
from flonacomldft.FES.plotter2 import Plotter

from flonacomldft.dft_utils import (
   get_path,
   Angles_transformation,
   AG6_construction_tables,
   Structure
)

from flonacomldft.adapt_mldft import run_mcmc

print("Done\n")

torch.manual_seed(42)
ceph_home = get_path()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)
print('Date:', date)

print('Device: %s...\n'%device)

print("Loading the database...")

df = pd.read_csv(ceph_home + 'is1_lcao_zmat.csv')

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

x_tensor = Angles_transformation(x_tensor)
x_tensor.inv_transf()

print("Done\n")

print("Normalizing flow training")

args_rnvp = {
    'dim': x.shape[1],
    'n_realnvp_block': 10,
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

ct1 = AG6_construction_tables('is1')

_ = run_mcmc(model, U_tensor, 
           x_tensor,
           n_iter=5000,
           lr=1e-1, #1e-3,
           bs=100,
           use_scheduler=False,
           step_schedule=100,
           args_loss={'type': 'fwd', 'samp': 'direct'},
           estimate_tau=False,
           return_all_xs=True,
           save_splits=10,
           grad_clip=1e4)

print("Done\n")

print("Plotting log_losses")

losses = torch.log(torch.tensor(_['losses']))
plt.figure(figsize=(7, 5))
plt.plot(losses)
plt.ylabel('Losses')
plt.grid()
plt.savefig('losses.png')

print("Done\n")

## Save all the training
print("Saving the models")
filename = 'mcmc'
outfile = open(filename,'wb')

mcmc = _
pickle.dump(mcmc, outfile)
outfile.close()

print("Done\n")

#print("Running metropolis")



#model = models[-1]
#x_init = x_tensor.clone()
#u_init = U_tensor.clone()
#n_steps = 1

#print(run_mcmc(model, 50, u_init, x_init, ct1, n_steps))

#print("Done\n")
