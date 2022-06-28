#from ase.parallel import parprint as print
import numpy as np

import gpaw.mpi as mpi
import torch
import pickle
import pandas as pd
import copy

from flonacomldft.dft_utils import (
    #Angles_transformation,
    Angles_mapping,
    Structure
)

from flonacomldft.sampling import run_metropolis

from flonacomldft.md_utils import (
    get_path,
    load_from_pickle,
    load_is_csv, 
    shuffle_arr,
    get_is1,
    get_is2,
    get_internal_coordinates,
    run_molecular_dynamics
)

from flonacomldft.mixture import Mixture

from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.train_from_data import train

ranks = np.arange(0, mpi.world.size)
rank = mpi.world.rank
comm = mpi.world.new_communicator(ranks)

num_seed = np.array([0])

if rank == 0:
    num_seed = np.array([np.random.randint(1e5)])
    print(rank, num_seed)
    
comm.broadcast(num_seed, 0)

print(rank, num_seed)
#torch.manual_seed(num_seed[0])
torch.manual_seed(42)

mpi.world.barrier()

M = Angles_mapping()

ceph_home = get_path()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

def init_model(mean, cov, device):
    args_rnvp = {
        'dim': cov.shape[0],
        'n_realnvp_block': 15,
        'block_depth': 1,
        # 'args_prior': {'type': 'standn'}, # standard Gaussian base
        'args_prior': {'type': 'white', 'cov': cov, 'mean': mean}, # Gaussian with non-trival mean and covariance for base
        'init_weight_scale': 1e-6,
        }

    model = RealNVP_MLP(args_rnvp['dim'], 
                    args_rnvp['n_realnvp_block'],
                    args_rnvp['block_depth'],
                    init_weight_scale=args_rnvp['init_weight_scale'],
                    prior_arg=args_rnvp['args_prior'],
                    device=device)

    return model

def run_NF(molecule, iterations, name, model=None, i=0):
    
    traj = run_molecular_dynamics(molecule, iterations, name, i)
    zmat = get_internal_coordinates(traj)

    print(zmat)

    u_tensor = zmat[:, -1]
    x_tensor = zmat[:, :-1]

    #print(x_tensor)
    cov = torch.cov(x_tensor.T)
    cov = torch.eye(12)*cov.mean()
    #print(cov)
    mean = x_tensor.mean(0)

    if name=='is1':
        count_tensor = torch.zeros(x_tensor.shape[0])
    elif name=='is2':
        count_tensor = torch.ones(x_tensor.shape[0])
    else:
        raise RuntimeError('Can not find isomer!')
    
    #x_tensor = Angles_transformation(x_tensor)
    #x_tensor.inv_transf()

    M.inv_mapping(x_tensor)

    if i==0:
        model = init_model(mean, cov)    
    else:
        model=model

    model_init = copy.deepcopy(model)

    _ = train(model, 
           x_tensor,
           n_iter=100,
           lr=5e-2,
           bs=10,
           use_scheduler=False,
           step_schedule=100,
           args_loss={'type': 'fwd', 'samp': 'direct'},
           estimate_tau=False,
           return_all_xs=True,
           save_splits=10,
           grad_clip=1e4)

    return x_tensor, u_tensor, count_tensor, _

xi_is1, ui_is1, ci_is1, _is1 = get_NF(get_is1(), 3, 'is1')
mpi.world.barrier()
xi_is2, ui_is2, ci_is2, _is2 = get_NF(get_is2(), 3, 'is2')

NF_is1 = _is1['models'][-1]
NF_is2 = _is2['models'][-1]

n_points = xi_is1.shape[0] + xi_is2.shape[0]

indexes = torch.randperm(n_points)

xis = shuffle_arr([xi_is1, xi_is2], indexes)
uis = shuffle_arr([ui_is1, ui_is2], indexes)
cis = shuffle_arr([ci_is1, ci_is2], indexes)

print(xis)

models = np.array([NF_is1, NF_is2])
mixture = Mixture(models, torch.tensor([0.75, 0.25]).detach())


j = 0
while j<3:

    n_sts = 3
    n_chains = 1

    mcmc = run_metropolis(model=mixture, u_init=uis, x_init=xis, count_init=cis, n_sample=n_chains, n_steps=n_sts, mixture=True)

    filename = 'metropolis_mix_'+str(n_chains)+'_'+str(n_sts)+'_'+str(j)+''
    outfile = open(filename, 'wb')
    pickle.dump(mcmc, outfile)
    outfile.close()

    mpi.world.barrier()

    ag6 = Structure()
    print('mcmc_real', mcmc[0])
    x_new = mcmc[0][-1, :]
    c_new = mcmc[-1][-1]
    #x_new = Angles_transformation(x_new)
    #x_new.transf()
    M.mapping(x_new)
    print('mh results',rank, x_new)
    ag6.build_zmat_matrix(x_new[0])
    is_ = ag6.molecule

    if c_new == 0:
        j=j+1
        xi_is1, ui_is1, ci_is1, _is1 = get_NF(is_, 2, 'is1', i=j)
        NF_is1 = _is['models'][-1]
    elif c_new == 1:
        j=j+1
        xi_is2, ui_is2, ci_is2, _is2 = get_NF(is_, 2, 'is2', i=j)
        NF_is1 = _is['models'][-1]
    else:
        raise RuntimeError('No valid isomer')

    n_points = xi_is1.shape[0] + xi_is2.shape[0]

    indexes = torch.randperm(n_points)

    xis = shuffle_arr([xi_is1, xi_is2], indexes)
    uis = shuffle_arr([ui_is1, ui_is2], indexes)
    cis = shuffle_arr([ci_is1, ci_is2], indexes)

    models = np.array([NF_is1, NF_is2])
    mixture = Mixture(models, torch.tensor([0.75, 0.25]).detach())

"""
models = _['models'][-1]

traj_is1 = run_molecular_dynamics(get_is1(), iters=2, name='is1', i=0)
zmat_is1 = get_internal_coordinates(traj_is1)

traj_is2 = run_molecular_dynamics(get_is2(), iters=2, name='is1', i=0)
zmat_is2 = get_internal_coordinates(traj_is2)


print(zmat_is1)

u_tensor_is1 = zmat_is1[:, -1]
x_tensor_is1 = zmat_is1[:, :-1]

u_tensor_is2 = zmat_is2[:, -1]
x_tensor_is2 = zmat_is2[:, :-1]

cov = torch.cov(x_tensor.T)
mean = x_tensor.mean(0)

x_tensor = Angles_transformation(x_tensor)
x_tensor.inv_transf()

args_rnvp = {
    'dim': x.shape[1],
    'n_realnvp_block': 15,
    'block_depth': 1,
    # 'args_prior': {'type': 'standn'}, # standard Gaussian base
    'args_prior': {'type': 'white', 'cov': cov, 'mean': mean}, # Gaussian with non-trival mean and covariance for base
    'init_weight_scale': 1e-6,
}

model = RealNVP_MLP(args_rnvp['dim'], 
                    args_rnvp['n_realnvp_block'],
                    args_rnvp['block_depth'],
                    init_weight_scale=args_rnvp['init_weight_scale'],
                    prior_arg=args_rnvp['args_prior'],
                    device=device)

model_init = copy.deepcopy(model)

_ = train(model, 
           x_tensor,
           n_iter=100,
           lr=5e-2,
           bs=10,
           use_scheduler=False,
           step_schedule=100,
           args_loss={'type': 'fwd', 'samp': 'direct'},
           estimate_tau=False,
           return_all_xs=True,
           save_splits=10,
           grad_clip=1e4)


models = _['models'][-1]

"""