#from ase.parallel import parprint as print
import numpy as np

import gpaw.mpi as mpi
import torch
import pickle
import pandas as pd

from flonacomldft.dft_utils import (
    Angles_transformation,
    Structure
)

import torch.backends.cudnn as cudnn

from flonacomldft.sampling import run_metropolis

from flonacomldft.md_utils import (
    get_path,
    load_from_pickle,
    load_is_csv, 
    shuffle_arr,
    get_is1,
    get_is2,
    get_internal_coordinates,
    run_molecular_dynamics
)

from flonacomldft.mixture import Mixture

from flonacomldft.train_from_data import train

ranks = np.arange(0, mpi.world.size)
rank = mpi.world.rank
comm = mpi.world.new_communicator(ranks)

num_seed = np.array([0])

if rank == 0:
    num_seed = np.array([np.random.randint(1e5)])
    
comm.broadcast(num_seed, 0)
torch.manual_seed(num_seed[0])
cudnn.deterministic = True
#torch.manual_seed(42)
import random
random.seed(num_seed[0])
np.random.seed(num_seed[0])

#print('Rank, Seed: ', rank, num_seed[0])

#mpi.world.barrier()

#print('Rank, number, 1: ', rank, torch.randperm(4))

ceph_home = get_path()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

from flonacomldft.training_utils import init_model, run_NF

#print('Rank, number, 2: ', rank, torch.randperm(4))


xi_is1, ui_is1, ci_is1, _is1 = run_NF(get_is1(), 2, 'is1', device, None, 0)
mpi.world.barrier()
xi_is2, ui_is2, ci_is2, _is2 = run_NF(get_is2(), 2, 'is2', device, None, 0)

#print(xi_is1, xi_is2)

NF_is1 = _is1['models'][-1]
NF_is2 = _is2['models'][-1]

#print('ranks, shapes', rank, xi_is1.shape, xi_is2.shape)

n_points = xi_is1.shape[0] + xi_is2.shape[0]

#print('npoints, rank', rank, n_points)
#print(n_points.shape)
indexes = torch.randperm(n_points)

print('rank, suffles', rank, indexes)

xis = shuffle_arr([xi_is1, xi_is2], indexes)
uis = shuffle_arr([ui_is1, ui_is2], indexes)
cis = shuffle_arr([ci_is1, ci_is2], indexes)

#print('rank, md', rank, xis)

models = np.array([NF_is1, NF_is2])
mixture = Mixture(models, torch.tensor([0.75, 0.25]).detach())

j = 0
while j<3:

    n_sts = 3
    n_chains = 1

    mcmc = run_metropolis(model=mixture, u_init=uis, x_init=xis, count_init=cis, n_sample=n_chains, n_steps=n_sts, mixture=True)

    filename = 'metropolis_mix_'+str(n_chains)+'_'+str(n_sts)+'_'+str(j)+''
    outfile = open(filename, 'wb')
    pickle.dump(mcmc, outfile)
    outfile.close()

    mpi.world.barrier()

    ag6 = Structure()
    print('mcmc_real', mcmc[0])
    x_new = mcmc[0][-1, :]
    c_new = mcmc[-1][-1]
    x_new = Angles_transformation(x_new)
    x_new.transf()
    print('mh results',rank, x_new)
    ag6.build_zmat_matrix(x_new[0])
    is_ = ag6.molecule

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

    indexes = torch.randperm(n_points)

    xis = shuffle_arr([xi_is1, xi_is2], indexes)
    uis = shuffle_arr([ui_is1, ui_is2], indexes)
    cis = shuffle_arr([ci_is1, ci_is2], indexes)

    models = np.array([NF_is1, NF_is2])
    mixture = Mixture(models, torch.tensor([0.75, 0.25]).detach())