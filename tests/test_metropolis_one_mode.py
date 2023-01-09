import torch
import numpy as np # TODO: avoid using numpy

import gpaw.mpi as mpi

from flonacomldft.utils.io_utils import (
    load_pickle_file,
    load_csv_file,
    save_pickle_file, 
    get_project_path
)

from flonacomldft.sampling import run_metropolis
from flonacomldft.models.mixture import Mixture

# set equal seed for all ranks for parallel computations

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

mode_label = 1 #or 2
# mcmc chains parameters

n_chains = 50
n_steps = 100
energy_type = 'mlp'

if mode_label==1:
    isomer = 0
elif mode_label==2:
    isomer = 1

xs = load_csv_file('is{:d}_lcao_zmat.csv'.format(mode_label))
xs = torch.cat( (xs, torch.full((xs.shape[0], 1), isomer)), dim=1 )

xs = xs[torch.randperm(xs.size()[0])]
xs = xs[:n_chains]

# configs to initialize the chains

x_init = xs[:, :12]
u_init = xs[:, 13]
isomer_init = xs[:, 14]

# flow models

flow_model = load_pickle_file('is{:d}_flow_dic_training.pkl'.format(mode_label))['model']

# mlp models

mlp_model = load_pickle_file('is{:d}_mlp_dic_training.pkl'.format(mode_label))['model']


# initialize metropolis simulation

out = run_metropolis(
    model=flow_model,
    x_init=x_init,
    u_init=u_init,
    isomer_init=isomer_init,
    n_chains=n_chains,
    n_steps=n_steps,
    n_run="",
    energy_type=energy_type,
    frac_dft=0.2,
    mlp_models=mlp_model,
    mixture=False,
    T=300,
    with_tqdm=True,
)

f = "experiments/mcmc_mode_{:d}_chains_{:d}_steps_{:d}.pkl".format(mode_label, n_chains, n_steps)
save_pickle_file(out, f, path=get_project_path())