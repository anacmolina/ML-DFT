import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.adapt import run_mcmc_adapt
from flonacomldft.gaussian_utils import (
    MoG, plot_2d_level
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype=torch.float32

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)

# Build parameters of test mixture
dim = 2
k = 2
means = []
covars = []
weights = []
cv = 1 * torch.eye(dim, dtype=dtype)
offset = 5

means_ = [torch.tensor([-9, -9], dtype=dtype),
        torch.tensor([-5, 5], dtype=dtype),
        torch.tensor([7, 1], dtype=dtype)]
 

for c in range(k):   
    means.append(means_[c])
    covars.append(cv)
    weights.append(1)

weights[0] = 2
covars[0][0,0] = 0.5
covars[0][1,1] = 0.5

mog = MoG(means, covars, weights=weights, dtype=dtype, device=device)

# Train

args_target = {
    'type': 'mog',
    'dim': mog.dim,
    'means': mog.means,
    'covars': mog.covars,
    'weights': mog.weights 
}

args_rnvp = {
    'dim': args_target['dim'],
    'n_realnvp_block': 2,
    'block_depth': 1,
    'args_prior': {'type': 'standn'},
    'init_weight_scale': 1e-6,
}

args_training = {
    'n_iter': int(1e2),
    'lr': 1e-2, 
    'bs': int(3e2)  # batchsize (will get # Langevin steps from bs and n_tot )
}

args_loss = {
        'samp': 'mhmalangevin', 'dt': 1e-4, 'beta': 1.0,
        'n_tot': 30,
        'n_steps_burnin': 1e2,
        'ratio_pos_init': None}

model = RealNVP_MLP(args_rnvp['dim'], args_rnvp['n_realnvp_block'],
                    args_rnvp['block_depth'],
                    init_weight_scale=args_rnvp['init_weight_scale'],
                    prior_arg=args_rnvp['args_prior'],
                    device=device)

model_init = copy.deepcopy(model)

x_init_samp = None

args_loss['x_init_samp'] = x_init_samp
_ = run_mcmc_adapt(model, mog, n_iter=args_training['n_iter'], 
            lr=args_training['lr'], bs=args_training['bs'],
            args_loss=args_loss
            )
to_return = _  
xs = to_return['xs']    
    
results = {
    'args_target': args_target,
    'args_model': args_rnvp, 
    'args_training': args_training,
    'target': mog,
    'model_init': model_init, 
    'model': model,
    'final_xs': xs[-1],
    'xs' : to_return['xs'],
    'models' : to_return['models'], 
    'losses': to_return['losses'],
    'acc_rates': to_return['acc_rates']
    
}

plt.figure(figsize=(12, 4))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
Us_rnvp = plot_2d_level(model_init, x_min=-10, x_max=10, n_points=100, ax=ax1)
Us_rnvp = plot_2d_level(model, x_min=-10, x_max=10, n_points=100, ax=ax2)
Us_g = plot_2d_level(mog, x_min=-10, x_max=10, n_points=100, ax=ax3)
plt.show(block=False)