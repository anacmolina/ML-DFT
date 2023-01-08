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

# mcmc chains parameters

n_chains = 10
n_steps = 50
energy_type = 'mlp'

# TODO: build this as a functions

xs_is1 = load_csv_file('is{:d}_lcao_zmat.csv'.format(1))
xs_is2 = load_csv_file('is{:d}_lcao_zmat.csv'.format(2))

xs_is1 = torch.cat( (xs_is1, torch.zeros((xs_is1.shape[0], 1)) ), dim=1 )
xs_is2 = torch.cat( (xs_is2, torch.ones((xs_is2.shape[0], 1)) ), dim=1 )

xs = torch.cat((xs_is1, xs_is2))
xs = xs[torch.randperm(xs.size()[0])]

xs = xs[:n_chains]

# configs to initialize the chains

x_init = xs[:, :12]
u_init = xs[:, 13]
isomer_init = xs[:, 14]

# flow models

flow_is1 = load_pickle_file('is1_flow_dic_training.pkl')
flow_is2 = load_pickle_file('is2_flow_dic_training.pkl')

flow_models = np.array([flow_is1['model'],
                      flow_is2['model']])

mixture = Mixture(flow_models, torch.tensor([0.5, 0.5]).detach())

# mlp models

mlp_is1 = load_pickle_file('is1_mlp_dic_training.pkl')
mlp_is2 = load_pickle_file('is2_mlp_dic_training.pkl')

mlp_models = np.array([mlp_is1['model'],
                      mlp_is2['model']])

# initialize metropolis simulation

out = run_metropolis(
    model=mixture,
    x_init=x_init,
    u_init=u_init,
    isomer_init=isomer_init,
    n_chains=n_chains,
    n_steps=n_steps,
    n_run="",
    energy_type=energy_type,
    frac_dft=0.2,
    model_mlps=mlp_models,
    mixture=True,
    T=300,
    with_tqdm=True,
)

f = "experiments/mcmc_chains_{:d}_steps{:d}_flow_dic_training.pkl".format(n_chains, n_steps)
save_pickle_file(out, f, path=get_project_path())