import pickle

import torch
import numpy as np
import pandas as pd

# import gpaw.mpi as mpi
from ase.parallel import parprint as print

# from flonacomldft.dft_utils import (
#     Structure
# )

from flonacomldft.utils.data_utils import (
    get_path,
    load_zmat_csv,
    load_from_pickle,
    save_pickle_file
)

from flonacomldft.utils.data_processing import get_mix_data

from flonacomldft.sampling import run_metropolis
from flonacomldft.models.mixture import Mixture

num_seed = np.array([0])
torch.manual_seed(42)

ceph_home = get_path()

is1 = load_zmat_csv('is1')
is2 = load_zmat_csv('is2')

xis, uis, cis = get_mix_data(is1, is2)

train_is1 = load_from_pickle(ceph_home + 'training_is1')
train_is2 = load_from_pickle(ceph_home + 'training_is2') 

models = np.array([train_is1['models'][-1],
                      train_is2['models'][-1]])

mixture = Mixture(models, torch.tensor([0.5, 0.5]).detach())

n_sts = 2
n_chains = 5

mlp_is1 = load_from_pickle(get_path() + 'mlp_is1')
mlp_is2 = load_from_pickle(get_path() + 'mlp_is2')

models_mlps = [mlp_is1, mlp_is2]


out = run_metropolis(model=mixture, u_init=uis[:n_chains], x_init=xis[:n_chains, :],
                   count_init=cis[:n_chains], n_chains=n_chains, n_steps=n_sts,
                   mixture=True, energy_type='mlp-dft', mlps=models_mlps,
                   with_tqdm=True)

# print(out['xs'])

filename = 'metropolis_mlp-dft_'+str(n_chains)+'_'+str(n_sts)+''
save_pickle_file(out, filename)

# print(_)
