import pickle

import matplotlib.pyplot as plt
import numpy as np

from flonacomldft.md_utils import (
    get_is1, 
    get_is2, 
    get_path,
    shuffle_arr,
    run_molecular_dynamics, 
    get_internal_coordinates
)

from flonacomldft.dft_utils import Angles_mapping, Structure
from flonacomldft.train_from_data import train

from flonacomldft.training_utils import run_NF
from flonacomldft.mixture import Mixture

from flonacomldft.real_nvp_mlp import RealNVP_MLP
import copy
import torch

from ase.io.trajectory import Trajectory

from flonacomldft.sampling import run_metropolis

ceph_home = get_path()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

torch.set_default_dtype(dtype)

def get_pos_energy(zmat):
    u_tensor = zmat[:, -1]
    x_tensor = zmat[:, :-1]
    return x_tensor, u_tensor

def run_md_get_zmat(molecule, iterations, file_name):
    traj = run_molecular_dynamics(molecule, iterations, file_name)
    zmat = get_internal_coordinates(traj).detach()
    return zmat

def init_model(zmat):
    
    #Set this as a default for all the files
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    x_tensor = zmat[:, :-1]
    
    cov = torch.cov(x_tensor.T)
    cov = torch.eye(12)*cov.mean()
    mean = x_tensor.mean(0)
    
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

def get_models(nf_):
    return [nf_[0]['models'][-1], nf_[1]['models'][-1]]

def run_nf(zmat, model):

    #do clone to copy and not touch the memory space
    x_tensor, u_tensor = get_pos_energy(zmat)

    M = Angles_mapping()
    M.inv_mapping(x_tensor)

    model_init = copy.deepcopy(model)

    _ = train(model, 
           x_tensor,
           n_iter=100,
           lr=5e-2,
           bs=100,
           use_scheduler=False,
           step_schedule=100,
           args_loss={'type': 'fwd', 'samp': 'direct'},
           estimate_tau=False,
           return_all_xs=True,
           save_splits=10,
           grad_clip=1e4)
    
    M.mapping(x_tensor)

    return _

def get_mix_data(data_1, data_2):
    xi_is1, ui_is1 = get_pos_energy(data_1)
    xi_is2, ui_is2 = get_pos_energy(data_2)
    ci_is1, ci_is2 = torch.zeros(xi_is1.shape[0]), torch.ones(xi_is2.shape[0]) 

    n_points = xi_is1.shape[0] + xi_is2.shape[0]

    indexes = torch.randperm(n_points)

    # Unifying all data from MD and shuffling in order to to Metropolis-Hastings
    xis = shuffle_arr([xi_is1, xi_is2], indexes)
    uis = shuffle_arr([ui_is1, ui_is2], indexes)
    cis = shuffle_arr([ci_is1, ci_is2], indexes)
    return xis, uis, cis
    
# Initial MCMC
md_iters = 2

# Get data
data_is1 = run_md_get_zmat(get_is1(), md_iters, 'is1_0')
data_is2 = run_md_get_zmat(get_is2(), md_iters, 'is2_0')

# Train initial NFs 
init_nf_is1 = run_nf(data_is1, init_model(data_is1))
init_nf_is2 = run_nf(data_is2, init_model(data_is2))

nf = [[init_nf_is1, init_nf_is2], ]

xis, uis, cis =  get_mix_data(data_is1, data_is2)
model_is1, model_is2 = get_models(nf[0])

models = [get_models(nf[0]), ]

n_sts = 3
n_chains = 1

mixture = Mixture(models[0], torch.tensor([0.75, 0.25]).detach())
xs_, accs_, us_, counts_ = run_metropolis(model=mixture, u_init=uis, x_init=xis, count_init=cis, n_sample=n_chains, n_steps=n_sts, mixture=True)

xs_acc = [xs_, ]
us_acc = [us_, ]
accs_s = [accs_, ] 
cs_acc = [counts_, ]

for i in range(2):
    data_acc = torch.cat((xs_acc[i].T, us_acc[i].reshape(n_sts,n_chains,1).T), dim=0).T
    
    print(data_acc)
    print(cs_acc[i].reshape(n_sts,n_chains,1))
    
    data_ex = data_acc.reshape(n_sts*n_chains, 13)
    
    cs_ex = cs_acc[i].reshape(n_sts*n_chains, 1) 
    n_is2 = cs_ex.sum(dim=0)
    cs_ex = torch.ones(data_ex.shape)*cs_ex
    
    #reduce dimensionality
    
    select = torch.ones(data_acc.shape)*cs_acc[i].reshape(n_sts,n_chains,1)
    
    is1_acc = data_acc[~select.bool()].reshape(n_sts*n_chains-n_is2.int(),13).unique(dim=0)
    is2_acc = data_acc[select.bool()].reshape(*n_is2.int(),13).unique(dim=0)
    
    if is1_acc.nelement!=0:
        data_is1 = torch.cat((data_is1, is1_acc))
    elif is2_acc.nelement!=0:
        data_is2 = torch.cat((data_is2, is2_acc))
    else:
        pass
    #print(is1_acc)
    #print(is2_acc)
    
    #break
    
    data_is1 = torch.cat((data_is1, is1_acc))
    data_is2 = torch.cat((data_is2, is2_acc))
    
    x_ = xs_acc[i].unique_consecutive(dim=0)[-1]
    c_ = cs_acc[i].unique_consecutive(dim=0)[-1]
    
    ag6 = Structure()
    print(x_[0])
    ag6.build_zmat_matrix(x_[0])
    
    if c_==0:
        name='is1'
    elif c_==1:
        name='is2'
    else:
        raise RuntimeError('Unknown value')
    
    md_data = run_md_get_zmat(ag6.molecule, md_iters, 'name_'+str(i+1))
    
    if c_==0:
        data_is1 = torch.cat((data_is1, md_data))
    elif c_==1:
        data_is2 = torch.cat((data_is2, md_data))                             
    else:
        pass
                             
    nf_is1 = run_nf(data_is1, models[i][0])
    nf_is2 = run_nf(data_is2, models[i][1])

    nf.append([nf_is1, nf_is2])
                             
    xis, uis, cis =  get_mix_data(data_is1, data_is2)
    model_is1, model_is2 = get_models(nf[i+1])
    models.append([model_is1, model_is2])
    
    mixture = Mixture(models[i+1], torch.tensor([0.75, 0.25]).detach())
    xs__, accs__, us__, counts__ = run_metropolis(model=mixture, u_init=uis, x_init=xis, count_init=cis, n_sample=n_chains, n_steps=n_sts, mixture=True)

    xs_acc.append(xs__)
    accs_s.append(accs__)
    us_acc.append(us__)                    
    cs_acc.append(counts__)


mcmc_md = [data_is1, data_is2, xs_acc, accs_s, us_acc, cs_acc]

filename = 'mcmc_md_'+str(n_chains)+'_'+str(n_sts)+''
outfile = open(filename, 'wb')
pickle.dump(mcmc_md, outfile)
outfile.close()