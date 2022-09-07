import pickle

import torch
import numpy as np

from ase.io.trajectory import Trajectory
import gpaw.mpi as mpi

from flonacomldft.mixture import Mixture
from flonacomldft.sampling import run_metropolis
from flonacomldft.dft_utils import Structure

from flonacomldft.internal_coordinates import (
    get_internal_coordinates,
    get_mix_data
)

from flonacomldft.data_utils import (
    get_path,
    load_zmat_csv,
    save_pickle_file,
    load_from_pickle
    )

from flonacomldft.md_mcmc import (
    run_nf,
    run_md_get_zmat
)

def get_models(nf_):
    return [nf_[0]['models'][-1], nf_[1]['models'][-1]]

# Seed initialization for random generations
ranks = np.arange(0, mpi.world.size)
rank = mpi.world.rank
comm = mpi.world.new_communicator(ranks)

num_seed = np.array([0])

if rank == 0:
    num_seed = np.array([np.random.randint(1e5)])
    
comm.broadcast(num_seed, 0)

print("Rank: %d \t Seed: %d"%(rank, num_seed[0]))
torch.manual_seed(num_seed[0])

mpi.world.barrier()

# From csv files previously generated
data_is1 = load_zmat_csv('is1')
data_is2 = load_zmat_csv('is2')

# Train initial NFs
init_nf_is1 = load_from_pickle(get_path() + 'training_is1')
init_nf_is2 = load_from_pickle(get_path() + 'training_is2')

nf = [[init_nf_is1, init_nf_is2], ]
models = [get_models(nf[0]), ]

# Shuffle data
xis, uis, cis =  get_mix_data(data_is1, data_is2)

n_runs = 10
n_chains = 1
n_sts = 3

# MD iterations
md_iters = 10

# Building the NF mixture model
mixture = Mixture(models[0], torch.tensor([0.5, 0.5]).detach())

# Running the initial MCMC
_ = run_metropolis( model=mixture, 
                    u_init=uis, 
                    x_init=xis, 
                    count_init=cis, 
                    n_sample=n_chains, 
                    n_steps=n_sts, 
                    mixture=True)

xs_acc = [_['xs'], ]
us_acc = [_['us'], ]
accs_s = [_['accs'], ] 
cs_acc = [_['counts'], ]

for i in range(n_runs):
    data_acc = torch.cat((xs_acc[i].T, us_acc[i].reshape(n_sts,n_chains,1).T), dim=0).T

    data_ex = data_acc.reshape(n_sts*n_chains, 13)
    
    cs_ex = cs_acc[i].reshape(n_sts*n_chains, 1) 
    n_is2 = cs_ex.sum(dim=0)
    cs_ex = torch.ones(data_ex.shape)*cs_ex

    select = torch.ones(data_acc.shape)*cs_acc[i].reshape(n_sts,n_chains,1)
    
    is1_acc = data_acc[~select.bool()].reshape(n_sts*n_chains-n_is2.int(),13).unique(dim=0)
    is2_acc = data_acc[select.bool()].reshape(*n_is2.int(),13).unique(dim=0)
    
    if is1_acc.nelement!=0:
        data_is1 = torch.cat((data_is1, is1_acc))
    elif is2_acc.nelement!=0:
        data_is2 = torch.cat((data_is2, is2_acc))
    else:
        pass

    x_ = None
    c_ = cs_acc[i].unique_consecutive(dim=0)[-1]
    
    if c_==0:
        name='is1'
        x_ = data_is1[-1, :-1]
    elif c_==1:
        name='is2'
        x_ = data_is1[-1, :-1]
    else:
        raise RuntimeError('Unknown value')
     
    ag6 = Structure()
    ag6.build_zmat_matrix(x_)
   
    mpi.world.barrier()
    if ((i+1)%2==0):
        md_data = run_md_get_zmat(ag6.molecule, md_iters, name+'_'+str(i+1), starting=False)
    
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
    __ = run_metropolis(model=mixture, u_init=uis, x_init=xis, count_init=cis, n_sample=n_chains, n_steps=n_sts, mixture=True)

    xs_acc.append(__['xs'])
    accs_s.append(__['accs'])
    us_acc.append(__['us'])                    
    cs_acc.append(__['counts'])

    filename_ = 'mcmc_md_'+str(i)+'_'+str(n_chains)+'_'+str(n_sts)
    save_pickle_file(__, filename_)

mcmc_md = {
    'data_is1': data_is1, 
    'data_is1': data_is2, 
    'xs_acc': xs_acc, 
    'accs_s': accs_s, 
    'us_acc': us_acc, 
    'cs_acc': cs_acc
    }

filename = 'mcmc_md_complete_'+str(n_runs)+'_'+str(n_chains)+'_'+str(n_sts)+''
save_pickle_file(mcmc_md, filename)