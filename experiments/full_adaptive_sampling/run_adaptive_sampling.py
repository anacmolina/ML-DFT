import torch
import numpy as np

from flonacomldft.utils.io_utils import (
    load_csv_file,
    load_pickle_file,
)

from flonacomldft.full_adaptive_sampling import adaptative_sampling

# energy_type="dft"
# energy_type="mlp-dft"
energy_type="mlp"

# mcmc params
n_runs = 15
n_chains = 5
n_steps = 3

flow_hyperparams_is1 = {'n_iter': 100,
    'lr': 5e-4,
    'use_scheduler': False,
    'step_schedule': 100,
    'save_splits': 10,
    'grad_clip': 1e4}

flow_hyperparams_is2 = {'n_iter': 100,
    'lr': 5e-4,
    'use_scheduler': False,
    'step_schedule': 100,
    'save_splits': 10,
    'grad_clip': 1e4}

mlp_hyperparams_is1 = {'n_iter': 100,
    'lr': 5e-2,
    'use_scheduler': False,
    'step_schedule': 100,
}

mlp_hyperparams_is2 = {'n_iter': 100,
    'lr': 5e-2,
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


mode_labels = [1, 2]

x_flow_train = torch.cat( [load_csv_file(
    'datasets/is{:d}_md_train.csv'.format(mode_label)
    ) for mode_label in mode_labels] ) 

x_flow_test = torch.cat( [load_csv_file(
    'datasets/is{:d}_md_test.csv'.format(mode_label)
    ) for mode_label in mode_labels] )

x_mlp_train = torch.cat( [x_flow_train.clone()] + [load_csv_file(
    'datasets/is{:d}_flow_train.csv'.format(mode_label)
    ) for mode_label in mode_labels] )

x_mlp_test = torch.cat( [x_flow_test.clone()] + [load_csv_file(
    'datasets/is{:d}_flow_test.csv'.format(mode_label)
    ) for mode_label in mode_labels] )


# TODO: shuffle again

# configurations for the running the adaptative sampling

# for flows
u_flow_train = x_flow_train.clone()[:, 13]
isomers_flow_train = x_flow_train.clone()[:, 14]
x_flow_train = x_flow_train.clone()[:, :12]

isomers_flow_test = x_flow_test.clone()[:, 14]
x_flow_test = x_flow_test.clone()[:, :12]

# for mlps
u_mlp_train = x_mlp_train.clone()[:, 13]
isomers_mlp_train = x_mlp_train.clone()[:, 14]
x_mlp_train = x_mlp_train.clone()[:, :12]

u_mlp_test = x_mlp_test.clone()[:, 13]
isomers_mlp_test = x_mlp_test.clone()[:, 14]
x_mlp_test = x_mlp_test.clone()[:, :12]

# pretrain flows and mlps

# loading pretrain flows models
init_nf_is1 = load_pickle_file("models/is1_flow_dic_training.pkl")
init_nf_is2 = load_pickle_file("models/is2_flow_dic_training.pkl")

# loading pretrain mlps models
init_mlp_is1 = load_pickle_file("models/is1_mlp_dic_training.pkl")
init_mlp_is2 = load_pickle_file("models/is2_mlp_dic_training.pkl")

init_flow_train = [init_nf_is1, init_nf_is2]
init_mlps = [init_mlp_is1, init_mlp_is2]


out = adaptative_sampling(
    xs_md_init_train=x_flow_train,
    us_md_init_train=u_flow_train,
    isomers_md_init_train=isomers_flow_train,
    xs_md_init_test=x_flow_test,
    isomers_md_init_test=isomers_flow_test,
    xs_dft_init_train=x_mlp_train,
    us_dft_init_train=u_mlp_train,
    isomers_dft_init_train=isomers_mlp_train,
    xs_dft_init_test=x_mlp_test,
    us_dft_init_test=u_mlp_test,
    isomers_dft_init_test=isomers_mlp_test,
    n_runs=n_runs,
    n_chains=n_chains,
    n_steps=n_steps,
    energy_type=energy_type,
    dict_flows_init=init_flow_train,
    flow_hyperparams=[flow_hyperparams_is1, flow_hyperparams_is2],
    mlp_hyperparams=[mlp_hyperparams_is1, mlp_hyperparams_is2],
    dict_mlps_init=init_mlps,
    retraining_mlp=True)

#f = "experiments/adaptative_sampling_runs_{:d}_chains_{:d}_steps_{:d}_flow_dic_training.pkl".format(n_runs, n_chains, n_steps)
#save_pickle_file(out, f, path=get_project_path())

