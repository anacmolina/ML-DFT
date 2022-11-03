from itertools import count
from ase.parallel import parprint as print

import torch
import numpy as np
import gpaw.mpi as mpi

from flonacomldft.data_utils import (
    get_path, 
    load_zmat_csv, 
    load_from_pickle
    )

from flonacomldft.internal_coordinates import Angles_mapping

from flonacomldft.mixture import Mixture, get_models
from flonacomldft.sampling import run_metropolis
from flonacomldft.internal_coordinates import get_mix_data
from flonacomldft.train_flow_from_data import (
    train_flow
    )

'''''
Parameters we should be able to play with:
number of steps of retraining flows/mlps
- lr of retraining
- number of MCMC steps in between retraining
- portion of the chains that will go through DFT 
''''' 


# Seed initialization for random generations
ranks = np.arange(0, mpi.world.size)
rank = mpi.world.rank
comm = mpi.world.new_communicator(ranks)

num_seed = np.array([0])

if rank == 0:
    num_seed = np.array([np.random.randint(1e5)])

comm.broadcast(num_seed, 0)

print("Rank: %d \t Seed: %d" % (rank, num_seed[0]))
# torch.manual_seed(num_seed[0])
torch.manual_seed(36)

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

# Adaptative MCMC

flow_train = [
    [init_nf_is1, init_nf_is2],
]
flow = [
    get_models(flow_train[0]),
]
mlps = [[init_mlp_is1, init_mlp_is2]]

weights = torch.tensor([0.5, 0.5]).detach()
mixture = Mixture(flow[0], weights)

n_runs = 3
n_chains = 100
n_sts = 5
energy_type = "mlp-dft" #"dft"  

# init sampling
_ = run_metropolis(
    model=mixture,
    u_init=uis[:n_chains],
    x_init=xis[:n_chains, :],
    count_init=cis[:n_chains],
    n_chains=n_chains,
    n_steps=n_sts,
    energy_type=energy_type,
    mlps=mlps[0],
    mixture=True,
)
""" print(_['xs'])
print(_['accs'])
print(_['ind_dft']) """

xs_acc = [
    _["xs"],
]
us_acc = [
    _["us"],
]
accs_s = [
    _["accs"],
]
cs_acc = [
    _["counts"],
]


def T(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


USE_DFT_ENERGIES = False
if energy_type == "dft" or energy_type == "mlp-dft":
    USE_DFT_ENERGIES = True

#TODO: falta coger las ultimas pasos de los chains para restart
#TODO: falta guardar la info del sampling
#TODO: falta actualizar los modelos in each run
# retrain flows
i = 0
for i in range(n_runs):
    data_for_flows = T(
        torch.cat(
            (
                T(xs_acc[i]),
                T(cs_acc[i].reshape(n_sts, n_chains, 1)),
            ),
            dim=0,
        )
    )

    #TODO: rewrite in one funtion the data for retrain flows and mlps
    #TODO: remove last column of the df you build when you have separate the mcmc chains

    data_for_flows = data_for_flows.reshape(n_sts * n_chains, data_for_flows.shape[-1])
    #data_for_flows = data_for_flows.unique(dim=0)

    mask_flow = data_for_flows[:, -1] == 1
    is1_prop = data_for_flows[~mask_flow][:, :-1]
    is2_prop = data_for_flows[mask_flow][:, :-1]

    # print(data_for_flows)
    #del data_for_flows
    #print(is1_prop)
    #print(is2_prop)

    M = Angles_mapping()
    M.inv_mapping(is1_prop)
    M.inv_mapping(is2_prop)

    new_flow_is1 = train_flow(
        init_nf_is1['model'],
        is1_prop,
        n_iter=100,
        lr=5e-3,
        bs=100,
        use_scheduler=False,
        step_schedule=100,
        args_loss={"type": "fwd", "samp": "direct"},
        save_splits=10,
        grad_clip=1e4,)


    new_flow_is2 = train_flow(
        init_nf_is2['model'],
        is2_prop,
        n_iter=100,
        lr=5e-3,
        bs=100,
        use_scheduler=False,
        step_schedule=100,
        args_loss={"type": "fwd", "samp": "direct"},
        save_splits=10,
        grad_clip=1e4,)

    #flow.append([new_flow_is1, new_flow_is2])

    # retrain MLPs
    if USE_DFT_ENERGIES:

        data_for_mlp = T(
            torch.cat(
                (
                    T(xs_acc[i]),
                    T(us_acc[i].reshape(n_sts, n_chains, 1)),
                    T(cs_acc[i].reshape(n_sts, n_chains, 1)),
                ),
                dim=0,
            )
        )

        #print(data_for_mlp, _['ind_dft'])
        data_for_mlp = data_for_mlp[_['ind_dft'].bool()]
        mask_mlp = data_for_mlp[:, -1] == 1

        is1_prop_dft = data_for_mlp[~mask_mlp]
        is2_prop_dft = data_for_mlp[mask_mlp]

        #print(data_for_mlp)
        #print(is1_prop_dft)
        #print(is2_prop_dft)

        if energy_type == "dft":
            # use all # all index must be  ind_dft == 1
            data_ex = data_for_mlp.reshape(n_sts * n_chains, data_for_mlp.shape[-1])

        elif energy_type == "mlp-dft":
            # use only ind_dft
            print("hi :(, T_T")  # _["ind_dft"])


    weights = torch.tensor([0.5, 0.5]).detach()
    mixture = Mixture([new_flow_is1['model'], new_flow_is2['model']], weights)

    _ = run_metropolis(
        model=mixture,
        u_init=uis[:n_chains],
        x_init=xis[:n_chains, :],
        count_init=cis[:n_chains],
        n_chains=n_chains,
        n_steps=n_sts,
        energy_type=energy_type,
        mlps=mlps[0],
        mixture=True,
    )

    import matplotlib.pyplot as plt
    plt.plot(_['accs'].mean(1))
    plt.show()