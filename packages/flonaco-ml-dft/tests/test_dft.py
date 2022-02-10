print('Starting...\n')

print("Loading the libraries...")
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import pandas as pd

from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.train_from_data import train
# from flonacomldft.adapt import run_mcmc_adapt

from flonacomldft.dft_utils import (
    Structure, 
    AG6_construction_tables,
    Angles_tranformation
    )
from flonacomldft.train_from_data import train

print("Done\n")

if os.path.isdir('/mnt/home/amolina/ceph/'):
   ceph_home = '/mnt/home/amolina/ceph/'
elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'):
   ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'
elif os.path.isdir('/home/anacristina/Monte Carlo Project/Trajectories/Data_4/'):
    ceph_home = '/home/anacristina/Monte Carlo Project/Trajectories/Data_4/'
else:
   raise RuntimeError('Data path not understood')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype=torch.float32

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)
print('Date:', date)

print('Device: %s...\n'%device)

print("Loading the database...")

data_home = ceph_home + 'Database/'

df = pd.read_csv(data_home + 'is1_zmat.csv')
df = df.drop(['Unnamed: 0'], axis=1)

print("Done\n")

print("Setting the input...")

U = df.energies#.to_numpy()
X = df.drop(['energies'], axis=1)#.to_numpy()

print("Done\n")

#DATA I'M CURRENTLY USING !REMEMBER
print("Loading the training data...")

n = -1
u = U[:n]
x = X[:n]

columns = x.columns.to_list()[6:]
x = x.apply(lambda y: np.arctan(y) if y.name in columns else y)

cov = torch.tensor(x.cov().to_numpy()).float()
mean = torch.tensor(x.describe().iloc[1]).float()

x_tensor = torch.from_numpy(x.to_numpy()).float()
U_tensor = torch.from_numpy(u.to_numpy()).float()

#x_tensor = Angles_tranformation(x_tensor)
#x_tensor.inv_transf()

# Compute mean and cov to put in base distribution of NF models
#mean = x_tensor.mean(0)
#cov = (x_tensor - mean).T @ (x_tensor - mean) / n 

#print(x_tensor.shape, U_tensor.shape)

#"""
print("Done\n")

print("Calculating the energy for one configuration")

i=0
symbols = np.full(6, 'Ag')
ct1 = AG6_construction_tables('is1')
ag6 = Structure(construction_table_=ct1, symbols_=symbols, Natoms_=len(symbols))

#ag6.calculate_potential_energy(x[i])
#pot_energy = ag6.potential_energy

print("Potential Energy: skipping")
#print(pot_energy, u[i], pot_energy-u[i])
#print(ag6.molecule.get_potential_energy(), 
# ag6.molecule.get_total_energy(),
# ag6.molecule.get_kinetic_energy())
#os.remove("ag6.out")
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
           x_tensor, ## to be replaced by something like Structure
           n_iter=50,
           lr=1e-1,
           bs=100,
           use_scheduler=False,
           step_schedule=100,
           args_loss={'type': 'fwd', 'samp': 'direct'},
           estimate_tau=False,
           return_all_xs=True,
           save_splits=10,
           grad_clip=1e4)

#"""

model = _['models'][0]

print(x_tensor[0])
print(model.sample(10))