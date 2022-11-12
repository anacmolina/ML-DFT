import torch
import numpy as np
import gpaw.mpi as mpi

from flonacomldft.data_utils import (
    get_path,
    load_zmat_csv,
    load_from_pickle,
    save_pickle_file,
)

from flonacomldft.internal_coordinates import Angles_mapping, get_mix_data

from flonacomldft.full_adaptative_sampling import adaptative_sampling

# Seed initialization for random generations
ranks = np.arange(0, mpi.world.size)
rank = mpi.world.rank
comm = mpi.world.new_communicator(ranks)

num_seed = np.array([0])

if rank == 0:
    num_seed = np.array([np.random.randint(1e5)])

comm.broadcast(num_seed, 0)

print("Rank: %d \t Seed: %d" % (rank, num_seed[0]))
torch.manual_seed(num_seed[0])
#torch.manual_seed(36)

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
init_nf_is1 = load_from_pickle(get_path() + "training_is1")
init_nf_is2 = load_from_pickle(get_path() + "training_is2")

init_flow_train = [init_nf_is1, init_nf_is2]
init_mlps = [init_mlp_is1, init_mlp_is2]

n_runs = 5
n_chains = 50
n_steps = 200

results = adaptative_sampling(
    xis[:n_chains],
    uis[:n_chains],
    cis[:n_chains],
    n_runs,
    n_chains,
    n_steps,
    "mlp",
    init_flow_train,
    init_mlps,
)

save_pickle_file(
    results,
    "runs_" + str(n_runs) + "_chains_" + str(n_chains) + "_steps_" + str(n_steps),
)
