import torch
import numpy as np

from flonacomldft.utils.io_utils import (
    load_csv_file,
    load_pickle_file,
    save_pickle_file,
    get_project_path
)

from flonacomldft.internal_coordinates import Coordinates_mapping, join_data
from flonacomldft.full_adaptive_sampling import adaptative_sampling

energy_type="dft"
# energy_type="mlp-dft"
# energy_type="mlp"

# mcmc params
n_runs = 1
n_chains = 5
n_steps = 2

# TODO: Check the mlp is1 training loss

flow_hyperparams_is0 = {'n_iter': 100,
    'lr': 1e-4,
    'use_scheduler': False,
    'step_schedule': 100,
    'save_splits': 10,
    'grad_clip': 1e4}

flow_hyperparams_is1 = {'n_iter': 100,
    'lr': 1e-4,
    'use_scheduler': False,
    'step_schedule': 100,
    'save_splits': 10,
    'grad_clip': 1e4}

mlp_hyperparams_is0 = {'n_iter': 100,
    'lr': 5e-5,
    'use_scheduler': False,
    'step_schedule': 100,
}

mlp_hyperparams_is1 = {'n_iter': 1000,
    'lr': 5e-5,
    'use_scheduler': False,
    'step_schedule': 100,
}


####### end of arguments and beginning of script

np_seed = 36
torch.manual_seed(np_seed)

# Seed initialization for random generations
if "dft" in energy_type:
    import gpaw.mpi as mpi
    ranks = np.arange(0, mpi.world.size)
    rank = mpi.world.rank
    comm = mpi.world.new_communicator(ranks)

    if rank == 0:
        num_seed = np.array([np_seed])
        # num_seed = np.array([np.random.randint(1e5)])

    comm.broadcast(num_seed, 0)
    print("Rank: %d \t Seed: %d" % (rank, num_seed[0]))
else:
    num_seed = np.array([np_seed])
    # num_seed = np.array([np.random.randint(1e5)])
    print("Seed: %d" % num_seed[0])


mode_labels = [0, 1]
dataset_labels = ['md_train', 'md_test', 'flow_train', 'flow_test']

coord_mapping = Coordinates_mapping()

zmat_datasets = []
xs_datasets = []

for dataset_label in dataset_labels:    
    zmats = []
    xs = []
    for mode_label in mode_labels:
        file='datasets/is{:d}_{:s}.csv'.format(mode_label, dataset_label)
        zmat = load_csv_file(file)
        
        x, logdetjac, energies = coord_mapping.get_real_centered_from_internal(zmat[:, :12],   
                                                                            isomer=mode_label,
                                                                            energies=zmat[:, 12],
                                                                            logdetjacs=zmat[:, 14])
        x = join_data(x, energies, zmat[:, 13], logdetjac)
        
        zmats.append(zmat)
        xs.append(x)
        
    zmats = torch.cat(zmats)
    xs = torch.cat(xs)

    zmat_datasets.append(zmats)
    xs_datasets.append(xs)

mlp_train = torch.cat([xs_datasets[i] for i in [0, 2]])
mlp_test = torch.cat([xs_datasets[i] for i in [1, 3]])

# pretrain flows and mlps

# flow models

flows_dic = [load_pickle_file('models/is{:d}_flow_dic_training.pkl'.format(mode_label)) for mode_label in mode_labels]

# mlp models

mlps_dic = [load_pickle_file('models/is{:d}_mlp_dic_training.pkl'.format(mode_label)) for mode_label in mode_labels]

out = adaptative_sampling(
    flow_init_train=xs_datasets[0],
    flow_init_test=xs_datasets[1],
    n_runs=n_runs,
    n_chains=n_chains,
    n_steps=n_steps,
    energy_type=energy_type,
    dict_flows_init=flows_dic,
    flow_hyperparams=[flow_hyperparams_is0, flow_hyperparams_is1],
    retraining_mlp=True,
    dict_mlps_init=mlps_dic,
    mlp_hyperparams=[mlp_hyperparams_is0, mlp_hyperparams_is1],
    mlp_init_train=mlp_train,
    mlp_init_test=mlp_test,
)

#f = "experiments/adaptative_sampling_runs_{:d}_chains_{:d}_steps_{:d}_flow_dic_training.pkl".format(n_runs, n_chains, n_steps)
#save_pickle_file(out, f, path=get_project_path())

