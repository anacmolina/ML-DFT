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

from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.train_from_data import train

from flonacomldft.dft_utils import *
from flonacomldft.train_from_data import train

from pathlib import Path
sys.path.insert(0,str(Path.home())+'/utils/python')
from plotter2 import Plotter

print("Done\n")

if os.path.isdir('/mnt/home/amolina/ceph/database'):
   ceph_home = '/mnt/home/amolina/ceph/database'
elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'):
   ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'
elif os.path.isdir('/home/anacristina/ml_dft_project/database/'):
    ceph_home = '/home/anacristina/ml_dft_project/database/'
elif os.path.isdir('/home/amolina/ml_dft_project/database/'):
    ceph_home = '/home/amolina/ml_dft_project/database/'
else:
   raise RuntimeError('Data path not understood')

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

print(x_tensor)

cov = torch.cov(x_tensor.T)
mean = x_tensor.mean(0)

x_tensor = Angles_tranformation(x_tensor)
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

_ = train(model, 
           x_tensor,
           n_iter=5000,
           lr=5e-3,
           bs=100,
           use_scheduler=False,
           step_schedule=100,
           args_loss={'type': 'fwd', 'samp': 'direct'},
           estimate_tau=False,
           return_all_xs=True,
           save_splits=10,
           grad_clip=1e4)

losses = _['losses']
plt.figure(figsize=(7, 5))
plt.plot(losses)
plt.ylabel('Losses')
plt.grid()
plt.show()

models = _['models']
N_samples=100

for i in range(len(models)):
   print("Model: ", i)
   x = models[i].sample(N_samples)
   x = Angles_tranformation(x)
   x.transf()
   x = x.clone().data.cpu().numpy()
   c_, r_ = get_CVs(x)

   ax = plotting_fes_db('is1', 'lcao')
   ax.plot(c_, r_, 'mo', label='Sample '+str(i)+'')
   ax.legend(loc='lower left', fontsize=20)
   #plt.show()
   plt.savefig('fe_model_'+str(i)+'.png')
