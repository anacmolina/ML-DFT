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
from flonacomldft.utils_plots import plot_2d_level

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype=torch.float32

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)

# Define a mixture of Gaussian (MoG) as the target rho_* - 
# Here we will try to sample the MoG with a 'mixture of flows'
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

#Cheating and putting the cov and mean of each component of th MOG in
# each Realnvp
args_rnvp_1 = {
    'dim': 2,
    'n_realnvp_block': 2,
    'block_depth': 1,
    'args_prior': {'type': 'white', 'cov': covars[0], 'mean': means[0]},
    'init_weight_scale': 1e-6,
}

args_rnvp_2 = {
    'dim': 2,
    'n_realnvp_block': 2,
    'block_depth': 1,
    'args_prior': {'type': 'white', 'cov': covars[1], 'mean': means[1]},
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

model1 = RealNVP_MLP(args_rnvp_1['dim'], args_rnvp_1['n_realnvp_block'],
                    args_rnvp_1['block_depth'],
                    init_weight_scale=args_rnvp_1['init_weight_scale'],
                    prior_arg=args_rnvp_1['args_prior'],
                    device=device)

model2 = RealNVP_MLP(args_rnvp_2['dim'], args_rnvp_1['n_realnvp_block'],
                    args_rnvp_2['block_depth'],
                    init_weight_scale=args_rnvp_1['init_weight_scale'],
                    prior_arg=args_rnvp_2['args_prior'],
                    device=device)



def sample(n, mix_weights, mix_flows, shuffle=True, return_mus=False):
    """
    number of samples : n
    mixture weights  - torch.array - shape (n_comp,)  : mix_weights
    mixture flows - list of realnvps - length n_comp   : mix_flows
    shuffle  : bool - to avoid that the algorithm orders the n samples per isomer
    """
    
    # choosing from which component each of the n_samples will be drawn
    a = torch.multinomial(mix_weights, n, replacement=True)
    
    # to avoid a long loop of size n, count how-many of each isomer to generate
    mus, counts = a.unique(return_counts=True)   

    xs = []
    cs = []
    for mu, count in zip(mus, counts):
        xs.append(mix_flows[mu].sample(count))
        cs += [mu.item()] * count

    # n samples with first those from isomer 1 and then those of isomer 2
    x = torch.cat(xs)

    if shuffle:
        perm = torch.randperm(n)
        x = x[perm, :]
        cs = torch.tensor(cs)[perm]

    if return_mus:
        return x, cs
    else:
        return x

def nll(x, mix_weights, mix_flows):
    args = []
    for model, weight in zip(mix_flows, mix_weights):
        args.append(- model.nll(x) + torch.log(weight))

    return - torch.logsumexp(torch.stack(args), 0)


n_points=1000 #grid for contour plot
x_min=-5
x_max=5
mix_weights = torch.tensor([0.25, 0.75])
mix_flows = [model1, model2]

log_prob = lambda x: - nll(x, mix_weights, mix_flows)

plt.figure()
ax = plt.subplot(111)
plot_2d_level(log_prob, x_min, x_max, n_points, ax=ax)

n_samples = 100
x = sample(n_samples, mix_weights, mix_flows)
ax.scatter(x[:,0], x[:,1], marker='.', color='k')
plt.show(block=False)
