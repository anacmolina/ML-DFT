'''
Script with all sampling methods. 

Computation of effective sampling size from:
https://github.com/jwalton3141/jwalton3141.github.io
following definition from:
ref Gelman, Andrew, J. B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin. 2013. Bayesian Data Analysis. Third Edition. London: Chapman & Hall / CRC Press.
'''

from flonacomldft.data_utils import (
    get_path,
    load_from_pickle
    )
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

def run_metropolis(model, u_init, x_init, count_init, n_chains, n_steps, energy_type=None, mixture=False):

    assert(u_init.shape[0]==n_chains)
    assert(x_init.shape[0]==n_chains)
    assert(count_init.shape[0]==n_chains)

    #print(u_init.shape, x_init.shape, count_init.shape)
    print('assert pass')

    xs = []
    accs = []
    us = []
    us_p = []
    nlls = []
    counts = []

    T=300
    kb = 8.617333262e-5
    beta = 1/(kb*T)

    M = Angles_mapping()

    for dt in range(n_steps):

        M.inv_mapping(x_init)
        
        if mixture:
            x, count = model.sample(n_chains, return_mus=True)
        else:
            x = model.sample(n_chains)
            count = count_init

        x = x.clone().detach().float()
        count = count.clone().detach().float()
    
        nll_x = model.nll(x)
        nll_x_init = model.nll(x_init)
    
        M.mapping(x)
        M.mapping(x_init)

        indexes_nc = None

        if energy_type == 'dft':
        
            #print('dft')

            ag6 = Structure()

            U_ = []
            indexes_nc = []

            for i in range(n_chains):
                try:
                    #ag6.calculate_potential_energy(x[i], txt='ag6_'+str(i)+'_'+str(dt)+'.out')
                    #U_.append(ag6.potential_energy)
                    U_.append(-6.3*(1+np.random.rand()*0.1))
                except:
                    U_.append(0)
                    indexes_nc.append(i)
            
            indexes_nc = torch.tensor(indexes_nc)
            U_ = torch.tensor(U_).float()
            U = U_.clone().detach().requires_grad_(True)

        elif energy_type == 'mlp':
        
            #print('mlps')

            U_ = torch.zeros((x.shape[0], 1))

            if(count.sum().int()==count.shape[0]):

                #print('mlp_is2')
                
                mlp_is2 = load_from_pickle(get_path()+"mlp_is2")
                model_mlp_is2 = Uncentered_MLP(mlp_is2)

                U_[count.bool()] = model_mlp_is2(x[count.bool()])

            if(count.sum().int()==0):

                #print('mlp_is1')

                mlp_is1 = load_from_pickle(get_path()+"mlp_is1")
                model_mlp_is1 = Uncentered_MLP(mlp_is1)
    
                U_[~(count.bool())] = model_mlp_is1(x[~(count.bool())])
            
            else:

                #print('mlp_is1_is2')

                mlp_is1 = load_from_pickle(get_path()+"mlp_is1")
                mlp_is2 = load_from_pickle(get_path()+"mlp_is2")

                model_mlp_is1 = Uncentered_MLP(mlp_is1)
                model_mlp_is2 = Uncentered_MLP(mlp_is2)

                U_[~(count.bool())] = model_mlp_is1(x[~(count.bool())])
                U_[count.bool()] = model_mlp_is2(x[count.bool()])
        
            U_ = U_.reshape(U_.shape[0]).float()
            U = U_.clone().float()

        elif energy_type == 'mlp-dft':
        
            #print('mlp-dft')
        
            U_ = torch.zeros((x.shape[0], 1))

            if(count.sum().int()==count.shape[0]):

                #print('mlp_is2')
                
                mlp_is2 = load_from_pickle(get_path()+"mlp_is2")
                model_mlp_is2 = Uncentered_MLP(mlp_is2)

                U_[count.bool()] = model_mlp_is2(x[count.bool()])

            if(count.sum().int()==0):

                #print('mlp_is1')

                mlp_is1 = load_from_pickle(get_path()+"mlp_is1")
                model_mlp_is1 = Uncentered_MLP(mlp_is1)
    
                U_[~(count.bool())] = model_mlp_is1(x[~(count.bool())])
            
            else:

                #print('mlp_is1_is2')

                mlp_is1 = load_from_pickle(get_path()+"mlp_is1")
                mlp_is2 = load_from_pickle(get_path()+"mlp_is2")

                model_mlp_is1 = Uncentered_MLP(mlp_is1)
                model_mlp_is2 = Uncentered_MLP(mlp_is2)

                U_[~(count.bool())] = model_mlp_is1(x[~(count.bool())])
                U_[count.bool()] = model_mlp_is2(x[count.bool()])
        
            U_ = U_.reshape(U_.shape[0]).float()

            n_dft = int(U_.shape[0]*0.2)

            if n_dft>0:

                U_sort, ind_U_sort = U_.sort()
    
                U_dft = []
                indexes_nc = []

                for x_ in x[ind_U_sort[:n_dft]]:
                    try:
                        #ag6.calculate_potential_energy(x_)
                        #U_dft.append(ag6.potential_energy)
                        U_dft.append(-6.3*(1+np.random.rand()*0.1))
                    except:
                        U_dft.append(0)
                        indexes_nc.append(i)
            
                U_[ind_U_sort[:n_dft]] = torch.tensor(U_dft).detach()

            U_ = U_.reshape(U_.shape[0]).float()
            U = U_.clone().float()

        else:
            raise RuntimeError('Unknown method for the energy')


        ratio =  - beta * (U) + nll_x
        ratio += beta * u_init - nll_x_init
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc = u < torch.min(ratio, torch.ones_like(ratio))

        indexes_nc = torch.tensor(indexes_nc)

        if(indexes_nc is not None and indexes_nc.shape[0] != 0):
            acc[indexes_nc] = torch.full((1, len(indexes_nc)), False)

        x[~acc] = x_init[~acc]
        U[~acc] = u_init[~acc]
    
        if mixture:
            count[~acc] = count_init[~acc]
        else:
            count = count_init

        xs.append(x.float().clone())
        accs.append(acc.float().clone())
        us.append(U.float().clone())
        us_p.append(U_.float().clone())
        nlls.append(nll_x.float().clone())
        counts.append(count.float().clone())

        x_init = x.clone().detach()
        u_init = U.clone().detach()
        count_init = count.clone().detach()

        print("acc: {:0.2f}".format(acc.float().mean()))

    to_return = {
        'xs': torch.stack(xs),
        'accs': torch.stack(accs),
        'us': torch.stack(us),
        'us_p': torch.stack(us_p),
        'counts': torch.stack(counts)
    }

    return to_return