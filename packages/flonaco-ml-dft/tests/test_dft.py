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
from flonacomldft.gaussian_utils import (
     MoG, plot_2d_level
 )

from flonacomldft.dft_utils import (
    Structure, 
    AG6_construction_tables
    )
from flonacomldft.train_from_data import train

print("Done\n")
#if os.path.isdir('/mnt/home/amolina/ceph/'):
#    ceph_home = '/mnt/home/amolina/ceph/'
#elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'):
#    ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'
#else:
#    raise RuntimeError('Data path not understood')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype=torch.float32

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)
print('Date:', date)

print('Device: %s...\n'%device)

print("Loading the database...")

ceph_home = '/home/anacristina/Monte Carlo Project/Trajectories/Data_4/'
data_home = ceph_home + 'Database/'

df = pd.read_csv(data_home + 'is1_zmat.csv')
df = df.drop(['Unnamed: 0'], axis=1)

print("Done\n")

print("Setting the input...")

U = df.energies.to_numpy()
X = df.drop(['energies'], axis=1).to_numpy()

print("Done\n")

#DATA I'M CURRENTLY USING !REMEMBER
print("Loading the training data...")

n = 200
u = U[:n]
x = X[:n]

x_tensor = torch.from_numpy(x).float()
U_tensor = torch.from_numpy(u).float()

#print(x_tensor.shape, U_tensor.shape)

#"""
print("Done\n")

print("Calculating the energy for one configuration")

i=0
symbols = np.full(6, 'Ag')
ct1 = AG6_construction_tables('is1')
ag6 = Structure(construction_table_=ct1, symbols_=symbols, Natoms_=len(symbols))

ag6.calculate_potential_energy(x[i])
pot_energy = ag6.potential_energy

print("Potential Energy")
print(pot_energy, u[i], pot_energy-u[i])
#print(ag6.molecule.get_potential_energy(), 
# ag6.molecule.get_total_energy(),
# ag6.molecule.get_kinetic_energy())
os.remove("ag6.out")
print("Done\n")

## code to apply necessary transforms (tanh -- MISSING )
##  and compute potential energy (Done)
## using the one here just for the test

print("Normalizing flow training")

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


# target = Struture(....)

model_init = copy.deepcopy(model)

class Probs:
    def __init__(self, U):
        #self.X_ = X
        self.U_ = U

    def U(self, x):
        return self.U_
    
target = Probs(U_tensor)
"""
dim = 1
k = 1
means = []
covars = []
weights = []
cv = 1 * torch.eye(dim, dtype=dtype)
offset = 5

means_ = [torch.tensor([U_tensor.mean()], dtype=dtype)]

for c in range(k):   
    means.append(means_[c])
    covars.append(cv)
    weights.append(1)

weights[0] = 2
covars[0][0,0] = U_tensor.std()**2

mog = MoG(means, covars, weights=weights, dtype=dtype, device=device)
"""
# train(.,....)

## 
_ = train(model, 
           x_tensor, ## to be replaced by something like Structure
           target=target,
           n_iter=200,
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