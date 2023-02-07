import torch
import numpy as np

import gpaw.mpi as mpi

from flonacomldft.utils.io_utils import (
    load_pickle_file,
    load_csv_file,
    save_pickle_file, 
    get_project_path
)

from flonacomldft.sampling import run_metropolis
from flonacomldft.models.mixture import Mixture
from flonacomldft.internal_coordinates import Coordinates_mapping

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

mode_labels = [0, 1]

# mcmc chains parameters

n_chains = 50
n_steps = 50
energy_type = 'mlp'

coord_mapping = Coordinates_mapping()

# load data

zmats_test = [load_csv_file("datasets/is{:d}_md_test.csv".format(mode_label)) for mode_label in mode_labels] 

xs = [coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 12],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 13]
                                    ) for mode_label, zmat_test in zip(mode_labels, zmats_test)]

xs = torch.stack([torch.cat((x[0], x[1].reshape(-1, 1), x[2].reshape(-1, 1), zmat_test[:, 14].reshape(-1, 1)), dim=1) for x, zmat_test in zip(xs, zmats_test)])
xs = xs.flatten(start_dim=0, end_dim=1).to(torch.float32)

# configs to initialize the chains

xs = xs[torch.randperm(xs.size()[0])]
xs = xs[:n_chains]

# flow models

flows_dic = [load_pickle_file('models/is{:d}_flow_dic_training.pkl'.format(mode_label)) for mode_label in mode_labels]
flow_models = np.array([flow_dic['model'] for flow_dic in flows_dic])

mixture = Mixture(flow_models, torch.tensor([0.5, 0.5]).detach())

# mlp models

mlps_dic = [load_pickle_file('models/is{:d}_mlp_dic_training.pkl'.format(mode_label)) for mode_label in mode_labels]
mlp_models = np.array([mlp_dic['model'] for mlp_dic in mlps_dic])


# initialize metropolis simulation

out = run_metropolis(
    model=mixture,
    init=xs,
    n_chains=n_chains,
    n_steps=n_steps,
    name_run="", # TODO: number of runs
    energy_type=energy_type,
    frac_dft=0.2,
    mlp_models=mlp_models,
    mixture=True,
    T=300,
    with_tqdm=True,
)

#f = "experiments/mcmc_mixture_chains_{:d}_steps_{:d}.pkl".format(n_chains, n_steps)
#save_pickle_file(out, f, path=get_project_path())
