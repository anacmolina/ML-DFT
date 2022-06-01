from ase.parallel import parprint as print
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

from flonacomldft.md_utils import (
    get_path,
    load_from_pickle,
    load_csv, 
    shuffle_arr,
    get_is1,
    get_is2,
    get_internal_coordinates,
    run_molecular_dynamics
)

from flonacomldft.mixture import Mixture

from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.train_from_data import train

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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

molecule = get_is2()
iterations = 3
name = 'is2'
i=0

traj = run_molecular_dynamics(molecule, iterations, name, i)
zmat = get_internal_coordinates(traj)

u_tensor = zmat[:, -1]
x_tensor = zmat[:, :-1]
print(x_tensor)
cov = torch.cov(x_tensor.T)
print(cov.min())
print(cov)
if cov.min()<0:
    cov = cov+(-1)*cov.min()

print(cov)