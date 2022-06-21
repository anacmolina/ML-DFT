#from ase.parallel import parprint as print
import numpy as np

import gpaw.mpi as mpi
import torch
import pickle
import pandas as pd
import random

# From this part of the library we import two clases
# Angles transformation is for applying tan and arctan to the angles
# Structure is a class to build the atom and calculate the energy 
from flonacomldft.dft_utils import (
    Angles_transformation,
    Structure
)

import torch.backends.cudnn as cudnn

# Run metropolis is a function to run Metropolis-Hastings
from flonacomldft.sampling import run_metropolis

# Importing the molecular structures
from flonacomldft.md_utils import (
    get_path,
    shuffle_arr,
    get_is1,
    get_is2,
)

# Class to build a mixture of two normalizing flows
from flonacomldft.mixture import Mixture

# Function to train the normalizing flow
from flonacomldft.train_from_data import train

# Function to run molecular dynamics and the normalizing flow training
from flonacomldft.training_utils import init_model, run_NF

# Setting the seed
ranks = np.arange(0, mpi.world.size)
rank = mpi.world.rank
comm = mpi.world.new_communicator(ranks)

num_seed = np.array([0])

if rank == 0:
    num_seed = np.array([np.random.randint(1e5)])

# Sharing the seed    
comm.broadcast(num_seed, 0)
torch.manual_seed(num_seed[0])
cudnn.deterministic = True

random.seed(num_seed[0])
np.random.seed(num_seed[0])

ceph_home = get_path()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

#Running MD and training the initial NF
xi_is1, ui_is1, ci_is1, _is1 = run_NF(get_is1(), 2, 'is1', device, None, 0)
mpi.world.barrier()
xi_is2, ui_is2, ci_is2, _is2 = run_NF(get_is2(), 2, 'is2', device, None, 0)

# Getting the las NF trained
NF_is1 = _is1['models'][-1]
NF_is2 = _is2['models'][-1]


n_points = xi_is1.shape[0] + xi_is2.shape[0]

indexes = torch.randperm(n_points)

print('rank, suffles', rank, indexes)

# Unifying all data from MD and shuffling in order to to Metropolis-Hastings
xis = shuffle_arr([xi_is1, xi_is2], indexes)
uis = shuffle_arr([ui_is1, ui_is2], indexes)
cis = shuffle_arr([ci_is1, ci_is2], indexes)

# Building the mixture
models = np.array([NF_is1, NF_is2])
mixture = Mixture(models, torch.tensor([0.75, 0.25]).detach())

# Loop to retrain one chain
j = 0
while j<3:

    n_sts = 3
    n_chains = 1

    # Metropolis-Hastings
    mcmc = run_metropolis(model=mixture, u_init=uis, x_init=xis, count_init=cis, n_sample=n_chains, n_steps=n_sts, mixture=True)

    #Saving information from MH
    filename = 'metropolis_mix_'+str(n_chains)+'_'+str(n_sts)+'_'+str(j)+''
    outfile = open(filename, 'wb')
    pickle.dump(mcmc, outfile)
    outfile.close()

    mpi.world.barrier()

    #Getting the last molecular structure of MH

    ag6 = Structure()
    x_new = mcmc[0][-1, :]
    c_new = mcmc[-1][-1]
    x_new = Angles_transformation(x_new)
    x_new.transf()
    ag6.build_zmat_matrix(x_new[0])
    is_ = ag6.molecule

    # Training a new NF with data that start from the last configurations of MH

    if c_new == 0:
        j=j+1
        xi_is1, ui_is1, ci_is1, _is1 = run_NF(is_, 2, 'is1', device,  models[0], i=j)
        NF_is1 = _is1['models'][-1]
    elif c_new == 1:
        j=j+1
        xi_is2, ui_is2, ci_is2, _is2 = run_NF(is_, 2, 'is2', device,  models[1], i=j)
        NF_is2 = _is2['models'][-1]
    else:
        raise RuntimeError('No valid isomer')

    n_points = xi_is1.shape[0] + xi_is2.shape[0]

    # Rebuilding the NF mixture to run MH again

    indexes = torch.randperm(n_points)

    xis = shuffle_arr([xi_is1, xi_is2], indexes)
    uis = shuffle_arr([ui_is1, ui_is2], indexes)
    cis = shuffle_arr([ci_is1, ci_is2], indexes)

    models = np.array([NF_is1, NF_is2])
    mixture = Mixture(models, torch.tensor([0.75, 0.25]).detach())