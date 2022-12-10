import torch
import numpy as np

from flonacomldft.utils.data_utils import (
    get_path,
    load_zmat_csv,
    load_from_pickle,
    save_pickle_file,
)
from flonacomldft.internal_coordinates import get_mix_data
from flonacomldft.full_adaptative_sampling import adaptative_sampling

energy_type="mlp-dft"

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
data_is1 = load_zmat_csv("is1")
data_is2 = load_zmat_csv("is2")

xis, uis, cis = get_mix_data(data_is1, data_is2)

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

flow_hyperparams = {'n_iter': 100,
    'lr': 5e-4,
    'use_scheduler': False,
    'step_schedule': 100,
    'save_splits': 10,
    'grad_clip': 1e4}

mlp_hyperparams = {'n_iter': 100,
    'lr': 5e-2,
    'use_scheduler': False,
    'step_schedule': 100,
    'grad_clip': 1e4,
}

results = adaptative_sampling(
    xs_md_init=xis,
    us_md_init=uis,
    isomers_md_init=cis,
    xs_dft_init=xis,
    us_dft_init=uis,
    isomers_dft_init=cis,
    n_runs=n_runs,
    n_chains=n_chains,
    n_steps=n_steps,
    energy_type=energy_type,
    dict_flows_init=init_flow_train,
    flow_hyperparams=flow_hyperparams,
    mlp_hyperparams=mlp_hyperparams,
    dict_mlps_init=init_mlps,
    retraining_mlp=True)

#save_pickle_file(
#    results,
#    "runs_" + str(n_runs) + "_chains_" + str(n_chains) + "_steps_" + str(n_steps),
#)
