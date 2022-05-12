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

from flonacomldft.files_utils import (
    get_path,
    load_from_pickle,
    load_csv, 
    shuffle_arr
)

from flonacomldft.mixture import Mixture

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

xi_is1, ui_is1, ci_is1 = load_csv('is1')
xi_is2, ui_is2, ci_is2 = load_csv('is2')

n_points = xi_is1.shape[0] + xi_is2.shape[0]


indexes = torch.randperm(n_points)

xis = shuffle_arr([xi_is1, xi_is2], indexes)
uis = shuffle_arr([ui_is1, ui_is2], indexes)
cis = shuffle_arr([ci_is1, ci_is2], indexes)

train_is1 = load_from_pickle(ceph_home + 'training_is1')
train_is2 = load_from_pickle(ceph_home + 'training_is2') 

models = np.array([train_is1['models'][-1],
                      train_is2['models'][-1]])

mixture = Mixture(models, torch.tensor([0.5, 0.5]).detach())

xi_is1 = Angles_transformation(xi_is1)
xi_is2 = Angles_transformation(xi_is2)
xis = Angles_transformation(xis)

#xi_is1.inv_transf()
#xi_is2.inv_transf()
xis.inv_transf()

n_sts = 2
n_chains = 2

#_ = run_metropolis(model=models[0], u_init=ui_is1, x_init=xi_is1, count_init=ci_is1, n_sample=100, n_steps=10, mixture=False)

#_ = run_metropolis(model=models[1], u_init=ui_is2, x_init=xi_is2, count_init=ci_is2, n_sample=100, n_steps=10, mixture=False)

_ = run_metropolis(model=mixture, u_init=uis, x_init=xis, count_init=cis, n_sample=n_chains, n_steps=n_sts, mixture=True)

filename = 'metropolis_mix_'+str(n_chains)+'_'+str(n_sts)+''
outfile = open(filename, 'wb')
pickle.dump(_, outfile)
outfile.close()
