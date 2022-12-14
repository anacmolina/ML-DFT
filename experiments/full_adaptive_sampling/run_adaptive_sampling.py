import torch
import numpy as np
import pandas as pd

from flonacomldft.utils.data_utils import (
    get_path,
    load_zmat_csv,
    load_from_pickle,
    save_pickle_file,
    split_data_from_dataframe
)

from flonacomldft.internal_coordinates import get_mix_data
from flonacomldft.full_adaptative_sampling import adaptative_sampling

#energy_type="dft"
energy_type="mlp-dft"
#energy_type="mlp"

# Seed initialization for random generations
if "dft" in energy_type:
    import gpaw.mpi as mpi
    ranks = np.arange(0, mpi.world.size)
    rank = mpi.world.rank
    comm = mpi.world.new_communicator(ranks)

    num_seed = np.array([36])

    if rank == 0:
        num_seed = np.array([36])
        # num_seed = np.array([np.random.randint(1e5)])

    comm.broadcast(num_seed, 0)
    print("Rank: %d \t Seed: %d" % (rank, num_seed[0]))
else:
    num_seed = np.array([36])
    # num_seed = np.array([np.random.randint(1e5)])
    print("Seed: %d" % num_seed[0])

torch.manual_seed(num_seed[0])

# Run MD for both isomers

# loading traj in internal coordinates

sk_seed = 42
train_size = 0.8
n_md = 2500 # 5000 steps in total
n_nf = 1500 # 2500 configs in total

df_md_is1 = pd.read_csv(get_path() + 'is1_lcao_zmat.csv').loc[:n_md]
x_train_md_is1, x_test_md_is1, y_train_md_is1, y_test_md_is1 = split_data_from_dataframe(df_md_is1, train_size, sk_seed)

df_nf_is1 = pd.read_csv(get_path() + 'x_nf_is1.csv').loc[:n_nf]
x_train_nf_is1, x_test_nf_is1, y_train_nf_is1, y_test_nf_is1 = split_data_from_dataframe(df_nf_is1, train_size, sk_seed)

df_md_is2 = pd.read_csv(get_path() + 'is2_lcao_zmat.csv').loc[:n_md]
x_train_md_is2, x_test_md_is2, y_train_md_is2, y_test_md_is2 = split_data_from_dataframe(df_md_is2, train_size, sk_seed)

df_nf_is2 = pd.read_csv(get_path() + 'x_nf_is2.csv').loc[:n_nf]
x_train_nf_is2, x_test_nf_is2, y_train_nf_is2, y_test_nf_is2 = split_data_from_dataframe(df_nf_is2, train_size, sk_seed)

x_train_flow = torch.cat((x_train_md_is1, x_train_md_is2))
y_train_flow = torch.cat((y_train_md_is1, y_train_md_is2))
isomers_train_flow = torch.cat((torch.zeros(x_train_md_is1.shape[0]), torch.ones(x_train_md_is2.shape[0])))  

x_test_flow = torch.cat((x_test_md_is1, x_test_md_is2))
y_test_flow = torch.cat((y_test_md_is1, y_test_md_is2))
isomers_test_flow = torch.cat((torch.zeros(x_test_md_is1.shape[0]), torch.ones(x_test_md_is2.shape[0])))

x_train_mlp = torch.cat((x_train_md_is1, x_train_nf_is1, x_train_md_is2, x_train_nf_is2))
y_train_mlp = torch.cat((y_train_md_is1, y_train_nf_is1, y_train_md_is2, y_train_nf_is2))
isomers_train_mlp = torch.cat((torch.zeros(x_train_md_is1.shape[0]+x_train_nf_is1.shape[0]), 
                                torch.ones(x_train_md_is2.shape[0]+x_train_nf_is2.shape[0])))  

x_test_mlp = torch.cat((x_test_md_is1, x_test_nf_is1, x_test_md_is2, x_test_nf_is2))
y_test_mlp = torch.cat((y_test_md_is1, y_test_nf_is1, y_test_md_is2, y_test_nf_is2))
isomers_test_mlp = torch.cat((torch.zeros(x_test_md_is1.shape[0]+x_test_nf_is1.shape[0]), 
                                torch.ones(x_test_md_is2.shape[0]+x_test_nf_is2.shape[0])))  

# Pretrain mlps and flows

# loading pretrain mlps models
init_mlp_is1 = load_from_pickle(get_path() + "mlp_is1")
init_mlp_is2 = load_from_pickle(get_path() + "mlp_is2")

# loading pretrain flows models
init_nf_is1 = load_from_pickle(get_path() + "flow_is1")
init_nf_is2 = load_from_pickle(get_path() + "flow_is2")

init_flow_train = [init_nf_is1, init_nf_is2]
init_mlps = [init_mlp_is1, init_mlp_is2]

# mcmc params
n_runs = 2
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
    'grad_clip': 1e4,
}

mlp_hyperparams_is2 = {'n_iter': 100,
    'lr': 5e-2,
    'use_scheduler': False,
    'step_schedule': 100,
    'grad_clip': 1e4,
}

results = adaptative_sampling(
    xs_md_init=x_train_flow,
    us_md_init=y_train_flow,
    isomers_md_init=isomers_train_flow,
    xs_md_init_test=x_test_flow,
    us_md_init_test=y_test_flow,
    isomers_md_init_test=isomers_test_flow,
    xs_dft_init=x_train_mlp,
    us_dft_init=y_train_mlp,
    isomers_dft_init=isomers_train_mlp,
    xs_dft_init_test=x_test_mlp,
    us_dft_init_test=y_test_mlp,
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

#save_pickle_file(
#    results,
#    "runs_" + str(n_runs) + "_chains_" + str(n_chains) + "_steps_" + str(n_steps),
#)
