import pickle
from datetime import datetime


import torch
import numpy as np

from ase.io.trajectory import Trajectory

from flonacomldft.mixture import Mixture, get_models
from flonacomldft.sampling import run_metropolis

from flonacomldft.data_utils import (
    get_path,
    load_zmat_csv,
    save_pickle_file,
    load_from_pickle,
)

from flonacomldft.dft_utils import get_internal_coordinates, Structure

from flonacomldft.md_mcmc import (
    init_model,
    get_mix_data,
    get_pos_energy,
    run_nf,
    run_md_get_zmat,
)

import gpaw.mpi as mpi

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

mpi.world.barrier()

# date = time.strftime('%d-%m-%Y')

# Loading previous trajectories
path = get_path()
traj_is1 = Trajectory(path + "ag6_is1_lcao.traj")
traj_is2 = Trajectory(path + "ag6_is2_lcao.traj")

# Getting internal coordinates of previous trajectories info
"""
From .traj files

data_is1 = get_internal_coordinates(traj_is1)
data_is2 = get_internal_coordinates(traj_is2) 
"""

# From csv files previously generated
data_is1 = load_zmat_csv("is1")
data_is2 = load_zmat_csv("is2")

# Train initial NFs
# init_nf_is1 = run_nf(data_is1, init_model(data_is1))
# init_nf_is2 = run_nf(data_is2, init_model(data_is2))
init_nf_is1 = load_from_pickle(get_path() + "training_is1")
init_nf_is2 = load_from_pickle(get_path() + "training_is2")
""" 
List to save info:

    - NFs trained
    - Models
"""
nf = [
    [init_nf_is1, init_nf_is2],
]
models = [
    get_models(nf[0]),
]

# Shuffle data
xis, uis, cis = get_mix_data(data_is1, data_is2)

"""
MCMC parameters:
    - n_runs: MCMC runs 
    - n_chains: MCMC chains
    - n_sts: MCMC steps
"""

n_runs = 10
n_chains = 1
n_sts = 200

# MD iterations
md_iters = 100

# Building the NF mixture model
mixture = Mixture(models[0], torch.tensor([0.75, 0.25]).detach())

# Running the initial MCMC
xs_, accs_, us_, counts_ = run_metropolis(
    model=mixture,
    u_init=uis,
    x_init=xis,
    count_init=cis,
    n_sample=n_chains,
    n_steps=n_sts,
    mixture=True,
)

"""
MCMC Results:
    - xs_acc: Configs mcmc 
    - us_acc: Energies mcmc
    - accs_s: Acceptance information
    - cs_acc: (State (0: Planar, 1: 3D)) Isomer configuration
"""

xs_acc = [
    xs_,
]
us_acc = [
    us_,
]
accs_s = [
    accs_,
]
cs_acc = [
    counts_,
]

# MCMC + MD

for i in range(n_runs):

    data_acc = torch.cat(
        (xs_acc[i].T, us_acc[i].reshape(n_sts, n_chains, 1).T), dim=0
    ).T

    data_ex = data_acc.reshape(n_sts * n_chains, 13)

    cs_ex = cs_acc[i].reshape(n_sts * n_chains, 1)
    n_is2 = cs_ex.sum(dim=0)
    cs_ex = torch.ones(data_ex.shape) * cs_ex

    select = torch.ones(data_acc.shape) * cs_acc[i].reshape(n_sts, n_chains, 1)

    is1_acc = (
        data_acc[~select.bool()]
        .reshape(n_sts * n_chains - n_is2.int(), 13)
        .unique(dim=0)
    )
    is2_acc = data_acc[select.bool()].reshape(*n_is2.int(), 13).unique(dim=0)

    if is1_acc.nelement != 0:
        data_is1 = torch.cat((data_is1, is1_acc))
    elif is2_acc.nelement != 0:
        data_is2 = torch.cat((data_is2, is2_acc))
    else:
        pass

    # x_ = None
    c_ = torch.randint(0, 2, (1,))  # cis[-1]

    if c_ == 0:
        name = "is1"
        x_ = data_is1[-1, :-1]
        ag6 = traj_is1[-1]
    elif c_ == 1:
        name = "is2"
        x_ = data_is1[-1, :-1]
        ag6 = traj_is2[-1]
    else:
        raise RuntimeError("Unknown value")

    # ag6.build_zmat_matrix(x_)

    mpi.world.barrier()
    if (i + 1) % 2 == 0:
        md_data = run_md_get_zmat(
            ag6, md_iters, name + "_" + str(i + 1), starting=False
        )

        if c_ == 0:
            data_is1 = torch.cat((data_is1, md_data))
        elif c_ == 1:
            data_is2 = torch.cat((data_is2, md_data))
        else:
            pass

    print("First flow")
    startTime = datetime.now().timestamp()
    nf_is1 = run_nf(data_is1, models[i][0])
    time_energy = datetime.now().timestamp() - startTime
    print("Time: %s" % time_energy)

    print("Second flow")
    startTime = datetime.now().timestamp()
    nf_is2 = run_nf(data_is2, models[i][1])
    time_energy = datetime.now().timestamp() - startTime
    print("Time: %s" % time_energy)

    nf.append([nf_is1, nf_is2])

    xis, uis, cis = get_mix_data(data_is1, data_is2)
    model_is1, model_is2 = get_models(nf[i + 1])
    models.append([model_is1, model_is2])

    mixture = Mixture(models[i + 1], torch.tensor([0.75, 0.25]).detach())
    xs__, accs__, us__, counts__ = run_metropolis(
        model=mixture,
        u_init=uis,
        x_init=xis,
        count_init=cis,
        n_sample=n_chains,
        n_steps=n_sts,
        mixture=True,
    )

    xs_acc.append(xs__)
    accs_s.append(accs__)
    us_acc.append(us__)
    cs_acc.append(counts__)

    filename_ = "mcmc_md_" + str(i) + "_" + str(n_chains) + "_" + str(n_sts)
    save_pickle_file([xs_acc, us_acc, accs_s, cs_acc], filename_)

mcmc_md = [data_is1, data_is2, xs_acc, accs_s, us_acc, cs_acc]

filename = (
    "mcmc_md_complete_" + str(n_runs) + "_" + str(n_chains) + "_" + str(n_sts) + ""
)
save_pickle_file(mcmc_md, filename)
