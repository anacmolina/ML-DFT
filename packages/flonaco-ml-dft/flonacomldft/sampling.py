'''
Script with all sampling methods. 

Computation of effective sampling size from:
https://github.com/jwalton3141/jwalton3141.github.io
following definition from:
ref Gelman, Andrew, J. B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin. 2013. Bayesian Data Analysis. Third Edition. London: Chapman & Hall / CRC Press.
'''

from operator import xor
from flonacomldft.models import Uncentered_MLP
import numpy as np
import torch

from ase.parallel import parprint as print
# from datetime import datetime
from flonacomldft.dft_utils import Structure
from flonacomldft.internal_coordinates import Angles_mapping


"""
def run_metropolis(model, target, x_init, n_steps):
    xs = []
    accs = []

    for dt in range(n_steps):
        x = model.sample(x_init.shape[0])
        ratio = - target.beta * target.U(x) + model.nll(x)
        ratio += target.beta * target.U(x_init) - model.nll(x_init)
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc = u < torch.min(ratio, torch.ones_like(ratio))
        x[~acc] = x_init[~acc]
        xs.append(x.clone())
        accs.append(acc)
        x_init = x.clone()

    return torch.stack(xs), torch.stack(accs)
"""

# MCMC we are running Metropolis-Hastings
def run_metropolis(model, u_init, x_init, count_init, n_sample, n_steps, mixture=False):
    
    import gpaw.mpi as mpi
    rank = mpi.world.rank

    xs = []
    accs = []
    us = []
    us_p = []
    nlls = []
    counts = []
    
    u_init=u_init[:n_sample]
    x_init=x_init[:n_sample]
    count_init=count_init[:n_sample]

    T=300
    kb = 8.617333262e-5
    beta = 1/(kb*T)

    M = Angles_mapping()

    for dt in range(n_steps):

        M.inv_mapping(x_init)
        
        if mixture:
            x, count = model.sample(n_sample, return_mus=True)
        else:
            x = model.sample(n_sample)
            count = count_init
                        
        x = torch.tensor(x).detach().float()
        count = torch.tensor(count).detach().float()
    
        nll_x = model.nll(x)
        nll_x_init = model.nll(x_init)
    
        M.mapping(x)
        M.mapping(x_init)
       
        ag6 = Structure()
        
        U_ = []
        indexes_nc = []
        for i in range(n_sample):
            try:
                #print('# Energy sample calculation: ', i)
                ag6.calculate_potential_energy(x[i], txt='ag6_'+str(i)+'_'+str(dt)+'.out')
                U_.append(ag6.potential_energy)
                #U_.append(-6.3*(1+np.random.rand()*0.1))
            except:
                #print("Error calculating the energy, adding 0 to keep the size. Sample: ", i)
                U_.append(0)
                indexes_nc.append(i)
    
        indexes_nc = torch.tensor(indexes_nc)
        U_ = torch.tensor(U_).float()
        U = U_.clone()
        ratio =  - beta * (U) + nll_x
        ratio += beta * u_init - nll_x_init
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc = u < torch.min(ratio, torch.ones_like(ratio))
    
        if(indexes_nc.shape[0] != 0):
            acc[indexes_nc] = torch.full((1, len(indexes_nc)), False)
    
        mpi.world.barrier()

        x[~acc] = x_init[~acc]
        U[~acc] = u_init[~acc]
    
        mpi.world.barrier()
        
        if mixture:
            count[~acc] = count_init[~acc]
        else:
            count = count_init
        
        mpi.world.barrier()
        xs.append(x.float().clone())
        accs.append(acc.float().clone())
        us.append(U.float().clone())
        us_p.append(U_.float().clone())
        nlls.append(nll_x.float().clone())
        counts.append(count.float().clone())
    
        mpi.world.barrier()
        x_init = x.clone().detach()
        u_init = U.clone().detach()
        count_init = count.clone().detach()
        
        mpi.world.barrier()
            
        print("Acceptance porcentage: %.1f%%"%(np.array(acc).sum()*100/len(acc)))

    to_return = {
        'xs': torch.stack(xs),
        'accs': torch.stack(accs),
        'us': torch.stack(us),
        'us_p': torch.stack(us_p),
        'counts': torch.stack(counts)
    }

    return to_return

# MCMC we are running Metropolis-Hastings and using MLPS energies
def run_metropolis_MLP(model, u_init, x_init, count_init, mlps, n_sample, n_steps, mixture=False):
    
    import gpaw.mpi as mpi
    rank = mpi.world.rank

    xs = []
    accs = []
    us = []
    us_p = []
    nlls = []
    counts = []
    
    u_init=u_init[:n_sample]
    x_init=x_init[:n_sample]
    count_init=count_init[:n_sample]

    T=300
    kb = 8.617333262e-5
    beta = 1/(kb*T)

    mlp_is1, mlp_is2 = mlps

    M = Angles_mapping()

    for dt in range(n_steps):

        M.inv_mapping(x_init)
        
        if mixture:
            x, count = model.sample(n_sample, return_mus=True)
        else:
            x = model.sample(n_sample)
            count = count_init
                        
        x = torch.tensor(x).detach().float()
        count = torch.tensor(count).detach().float()
    
        nll_x = model.nll(x)
        nll_x_init = model.nll(x_init)
    
        M.mapping(x)
        M.mapping(x_init)
       
        model_mlp_is1 = Uncentered_MLP(mlp_is1)
        model_mlp_is2 = Uncentered_MLP(mlp_is2)

        U_ = torch.zeros((x.shape[0], 1))

        if(count.sum().int()==count.shape[0]):
            U_[count.bool()] = model_mlp_is2(x[count.bool()])
        if(count.sum().int()==0):
            U_[~(count.bool())] = model_mlp_is1(x[~(count.bool())])
        else:
            U_[~(count.bool())] = model_mlp_is1(x[~(count.bool())])
            U_[count.bool()] = model_mlp_is2(x[count.bool()])
        
        U = U_.clone()
        U = U.reshape(U_.shape[0])

        ratio =  - beta * (U) + nll_x
        ratio += beta * u_init - nll_x_init
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc = u < torch.min(ratio, torch.ones_like(ratio))
    
        mpi.world.barrier()

        x[~acc] = x_init[~acc]
        U[~acc] = u_init[~acc]
    
        mpi.world.barrier()
        
        if mixture:
            count[~acc] = count_init[~acc]
        else:
            count = count_init
        
        mpi.world.barrier()
        xs.append(x.float().clone())
        accs.append(acc.float().clone())
        us.append(U.float().clone())
        us_p.append(U_.float().clone())
        nlls.append(nll_x.float().clone())
        counts.append(count.float().clone())
    
        mpi.world.barrier()
        x_init = x.clone().detach()
        u_init = U.clone().detach()
        count_init = count.clone().detach()
        
        mpi.world.barrier()
            
        print("Acceptance porcentage: %.1f%%"%(np.array(acc).sum()*100/len(acc)))

    to_return = {
        'xs': torch.stack(xs),
        'accs': torch.stack(accs),
        'us': torch.stack(us),
        'us_p': torch.stack(us_p),
        'counts': torch.stack(counts)
    }

    return to_return
