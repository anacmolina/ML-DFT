import torch
from flonacomldft.real_nvp_mlp import RealNVP_MLP
import copy
from flonacomldft.train_from_data import train

from flonacomldft.md_utils import (
    get_is1,
    get_is2,
    get_internal_coordinates,
    run_molecular_dynamics
)

from flonacomldft.dft_utils import (
    Angles_transformation
)

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

def run_NF(molecule, iterations, name, device, model, i):
    
    import gpaw.mpi as mpi
    rank = mpi.world.rank

    traj = run_molecular_dynamics(molecule, iterations, name, i)
    mpi.world.barrier()
    #print('traj', rank, len(traj))
    zmat = get_internal_coordinates(traj)

    #print('zmat ', rank, zmat, zmat.shape)

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
    
    x_tensor = Angles_transformation(x_tensor)
    x_tensor.inv_transf()

    if i==0:
        model = init_model(mean, cov, device)    
    else:
        model=model

    model_init = copy.deepcopy(model)

    _ = train(model, 
           x_tensor,
           n_iter=10000,
           lr=5e-3,
           bs=100,
           use_scheduler=False,
           step_schedule=100,
           args_loss={'type': 'fwd', 'samp': 'direct'},
           estimate_tau=False,
           return_all_xs=True,
           save_splits=10,
           grad_clip=1e4)

    return x_tensor, u_tensor, count_tensor, _
