import torch
import numpy as np
from flonacomldft.utils.data_utils import (
    get_path,
    load_zmat_csv,
    load_from_pickle,
    # save_pickle_file
)
from flonacomldft.internal_coordinates import get_mix_data
from flonacomldft.sampling import run_metropolis
from flonacomldft.mixture import Mixture
from flonacomldft.free_energy_computations import compute_BAR, compute_TFP

torch.manual_seed(42)
ceph_home = get_path()

is1 = load_zmat_csv('is1')
is2 = load_zmat_csv('is2')

xis, uis, cis = get_mix_data(is1, is2)

dic_flow_training_is1 = load_from_pickle(ceph_home + 'training_is1')
dic_flow_training_is2 = load_from_pickle(ceph_home + 'training_is2') 

models = np.array([dic_flow_training_is1['model'],
                   dic_flow_training_is2['model']])
flow_is1 = models[0]
flow_is2 = models[1]

mixture = Mixture(models, torch.tensor([0.5, 0.5]).detach())

n_steps = 200
n_chains = 10

mlp_is1 = load_from_pickle(get_path() + 'mlp_is1')
mlp_is2 = load_from_pickle(get_path() + 'mlp_is2')

models_mlps = [mlp_is1, mlp_is2]

out = run_metropolis(model=mixture, u_init=uis[:n_chains], x_init=xis[:n_chains, :],
                   count_init=cis[:n_chains], n_chains=n_chains, n_steps=n_steps,
                   mixture=True, energy_type='mlp', mlps=models_mlps,
                   with_tqdm=True)

xs1 = out['xs'][out['counts'] == 0].detach()
xs2 = out['xs'][out['counts'] == 1].detach()

logr_bar1, logr_err_bar1 = compute_BAR(xs1, target_log_prob=lambda x: - mlp_is1(x),
                                       prop=flow_is1)

n_prop = xs1.shape[0]
logr_tfp1, logr_err_tfp1 = compute_TFP(n_prop, target_log_prob=lambda x: - mlp_is1(x),
                                       prop=flow_is1)

print('logr_bar1: ', logr_bar1, 'logr_err_bar1: ', logr_err_bar1)
print('logr_tfp1: ', logr_tfp1, 'logr_err_tfp1: ', logr_err_tfp1)

# print(out['xs'])

# filename = 'metropolis_mlp-dft_'+str(n_chains)+'_'+str(n_steps)+''
# save_pickle_file(out, filename)

# print(_)
