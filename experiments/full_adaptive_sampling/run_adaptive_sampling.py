import torch
import numpy as np

from flonacomldft.utils.io_utils import (
    get_project_path,
    load_csv_file,
    load_pickle_file,
    save_pickle_file,
)

from flonacomldft.utils.data_processing import split_data_from_dataframe
from flonacomldft.full_adaptive_sampling import adaptative_sampling


# energy_type="dft"
# energy_type="mlp-dft"
energy_type="mlp"

# mcmc params
n_runs = 10
n_chains = 50
n_steps = 30

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

np_seed = 36
sk_seed = 42 # BE CAREFUL! same for all the splitting data in flows and mlps
train_size = 0.8
n_md = 2500 # 5000 steps in total
n_flow = 20 # 2500 configs in total

####### end of arguments and beginning of script

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


xs_md_is1 = load_csv_file('is1_lcao_zmat.csv')[:n_md]
xs_flow_is1 = load_csv_file('is1_flow_zmat.csv')[:n_flow]
xs_md_is2 = load_csv_file('is2_lcao_zmat.csv')[:n_md]
xs_flow_is2 = load_csv_file('is2_flow_zmat.csv')[:n_flow]

# add column with 0 for is1 configs and 1 for is2 configs
xs_md_is1 = torch.cat( (xs_md_is1, torch.zeros((xs_md_is1.shape[0], 1)) ), dim=1 )
xs_flow_is1 = torch.cat( (xs_flow_is1, torch.zeros((xs_flow_is1.shape[0], 1)) ), dim=1 )
xs_md_is2 = torch.cat( (xs_md_is2, torch.ones((xs_md_is2.shape[0], 1)) ), dim=1 )
xs_flow_is2 = torch.cat( (xs_flow_is2, torch.ones((xs_flow_is2.shape[0], 1)) ), dim=1 )

# splitting data into train and test
x_train_md_is1, x_test_md_is1 = split_data_from_dataframe(xs_md_is1, train_size, sk_seed)
x_train_flow_is1, x_test_flow_is1 = split_data_from_dataframe(xs_flow_is1, train_size, sk_seed)
x_train_md_is2, x_test_md_is2 = split_data_from_dataframe(xs_md_is2, train_size, sk_seed)
x_train_flow_is2, x_test_flow_is2 = split_data_from_dataframe(xs_flow_is2, train_size, sk_seed)

x_train_flow = torch.cat((x_train_md_is1, x_train_md_is2))
x_test_flow = torch.cat((x_test_md_is1, x_test_md_is2))

x_train_mlp = torch.cat((x_train_md_is1, x_train_flow_is1, x_train_md_is2, x_train_flow_is2))
x_test_mlp = torch.cat((x_test_md_is1, x_test_flow_is1, x_test_md_is2, x_test_flow_is2))

# configurations for the running the adaptative sampling

# for flows
u_train_flow = x_train_flow[:, 13]
isomers_train_flow = x_train_flow[:, 14]
x_train_flow = x_train_flow[:, :12]

isomers_test_flow = x_test_flow[:, 14]
x_test_flow = x_test_flow[:, :12]

# for mlps
u_train_mlp = x_train_mlp[:, 13]
isomers_train_mlp = x_train_mlp[:, 14]
x_train_mlp = x_train_mlp[:, :12]

u_test_mlp = x_test_mlp[:, 13]
isomers_test_mlp = x_test_mlp[:, 14]
x_test_mlp = x_test_mlp[:, :12]

# pretrain flows and mlps

# loading pretrain flows models
init_nf_is1 = load_pickle_file("is1_flow_dic_training.pkl")
init_nf_is2 = load_pickle_file("is2_flow_dic_training.pkl")

# loading pretrain mlps models
init_mlp_is1 = load_pickle_file("is1_mlp_dic_training.pkl")
init_mlp_is2 = load_pickle_file("is2_mlp_dic_training.pkl")

init_flow_train = [init_nf_is1, init_nf_is2]
init_mlps = [init_mlp_is1, init_mlp_is2]


out = adaptative_sampling(
    xs_md_init_train=x_train_flow,
    us_md_init_train=u_train_flow,
    isomers_md_init_train=isomers_train_flow,
    xs_md_init_test=x_test_flow,
    isomers_md_init_test=isomers_test_flow,
    xs_dft_init_train=x_train_mlp,
    us_dft_init_train=u_train_mlp,
    isomers_dft_init_train=isomers_train_mlp,
    xs_dft_init_test=x_test_mlp,
    us_dft_init_test=u_test_mlp,
    isomers_dft_init_test=isomers_test_mlp,
    n_runs=n_runs,
    n_chains=n_chains,
    n_steps=n_steps,
    energy_type=energy_type,
    dict_flows_init=init_flow_train,
    flow_hyperparams=[flow_hyperparams_is1, flow_hyperparams_is2],
    mlp_hyperparams=[mlp_hyperparams_is1, mlp_hyperparams_is2],
    dict_mlps_init=init_mlps,
    retraining_mlp=True)

f = "experiments/adaptative_sampling_runs_{:d}_chains_{:d}_steps_{:d}_flow_dic_training.pkl".format(n_runs, n_chains, n_steps)
save_pickle_file(out, f, path=get_project_path())

