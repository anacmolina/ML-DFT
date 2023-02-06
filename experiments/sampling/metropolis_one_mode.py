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

mode_label = 0 #or 1

# mcmc chains parameters
n_chains = 50
n_steps = 1
energy_type = 'mlp'

coord_mapping = Coordinates_mapping()

# loading data
zmat_test = load_csv_file("datasets/is{:d}_md_test.csv".format(mode_label)) 

xs_test, logdetjacs_test, energies_test = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 12],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 13]
                                    )

xs = torch.cat((xs_test, logdetjacs_test.reshape(-1, 1), energies_test.reshape(-1, 1), zmat_test[:, 14].reshape(-1, 1)), dim=1).to(torch.float32)

# configs to initialize the chains
xs = xs[torch.randperm(xs.size()[0])]
xs = xs[:n_chains]

# flow models
flow_model = load_pickle_file('models/is{:d}_flow_dic_training.pkl'.format(mode_label))['model']

# mlp models
mlp_model = load_pickle_file('models/is{:d}_mlp_dic_training.pkl'.format(mode_label))['model']


# initialize metropolis simulation

out = run_metropolis(
    model=flow_model,
    init=xs,
    n_chains=n_chains,
    n_steps=n_steps,
    name_run="",
    energy_type=energy_type,
    frac_dft=0.2,
    mlp_models=mlp_model,
    mixture=False,
    T=300,
    with_tqdm=True,
)

#f = "experiments/mcmc_mode_{:d}_chains_{:d}_steps_{:d}.pkl".format(mode_label, n_chains, n_steps)
#save_pickle_file(out, f, path=get_project_path())