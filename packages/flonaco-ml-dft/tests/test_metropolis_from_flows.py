import pickle

import torch
import numpy as np
import pandas as pd

import gpaw.mpi as mpi
from ase.parallel import parprint as print

from flonacomldft.dft_utils import (
    Structure
)

from flonacomldft.utils.data_utils import (
    get_path,
    load_zmat_csv,
    load_from_pickle,
    save_pickle_file
)

from flonacomldft.internal_coordinates import (
    shuffle_arr,
    get_mix_data
)

from flonacomldft.sampling import run_metropolis
from flonacomldft.models.mixture import Mixture

ranks = np.arange(0, mpi.world.size)
rank = mpi.world.rank
comm = mpi.world.new_communicator(ranks)

num_seed = np.array([0])

if rank == 0:
    num_seed = np.array([np.random.randint(1e5)])
    print(rank, num_seed)
    
comm.broadcast(num_seed, 0)

print(rank, num_seed)
torch.manual_seed(num_seed[0])

mpi.world.barrier()

ceph_home = get_path()

is1 = load_zmat_csv('is1')
is2 = load_zmat_csv('is2')

xis, uis, cis = get_mix_data(is1, is2)

train_is1 = load_from_pickle(ceph_home + 'training_is1')
train_is2 = load_from_pickle(ceph_home + 'training_is2') 

models = np.array([train_is1['models'][-1],
                      train_is2['models'][-1]])

mixture = Mixture(models, torch.tensor([0.5, 0.5]).detach())

n_sts = 5
n_chains = 3

#_ = run_metropolis(model=models[0], u_init=ui_is1, x_init=xi_is1, count_init=ci_is1, n_sample=n_chains, n_steps=n_sts, mixture=False)
#_ = run_metropolis(model=models[1], u_init=ui_is2, x_init=xi_is2, count_init=ci_is2, n_sample=n_chains, n_steps=n_sts, mixture=False)
_ = run_metropolis(model=mixture, u_init=uis, x_init=xis, count_init=cis, n_sample=n_chains, n_steps=n_sts, mixture=True)

filename = 'metropolis_mix_'+str(n_chains)+'_'+str(n_sts)+''
save_pickle_file(_, filename)

#print(_)
