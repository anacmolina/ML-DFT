import torch
import pandas as pd
import tqdm

import matplotlib.pyplot as plt

from flonacomldft.utils.io_utils import (
    get_path, 
    save_pickle_file, 
    split_data_from_dataframe,
    get_project_path
)

from flonacomldft.models.mlp import MLP
from flonacomldft.train_mlp_from_data import train_mlp
from flonacomldft.internal_coordinates import logdetjac_to_xyz, Structure

isomer = 'is1'

sk_seed = 42
train_size = 0.8
kb = 8.617333262e-5
T = 300

n_md = 200 # 5000 steps in total
df_md = pd.read_csv(get_path() + isomer + '_lcao_zmat.csv').loc[:n_md]

x_train_md, x_test_md, u_train_md, u_test_md = split_data_from_dataframe(df_md, train_size, sk_seed)

n_nf = 100 # 2500 configs in total
df_nf = pd.read_csv(get_path() + 'x_nf_'+ isomer +'.csv').loc[:n_nf]

x_train_nf, x_test_nf, u_train_nf, u_test_nf = split_data_from_dataframe(df_nf, train_size, sk_seed)

x_train = torch.cat((x_train_md, x_train_nf))
u_train = torch.cat((u_train_md, u_train_nf))
x_test = torch.cat((x_test_md, x_test_nf))
u_test = torch.cat((u_test_md, u_test_nf))

structure = Structure()
u_train_wjac = []
for u, x in tqdm.tqdm(zip(u_train, x_train)):
    u_wjac = u - (kb * T) * logdetjac_to_xyz(x, structure)[1]
    u_train_wjac.append(u_wjac)

u_test_wjac = []
for u, x in tqdm.tqdm(zip(u_test, x_test)):
    u_wjac = u - (kb * T) * logdetjac_to_xyz(x, structure)[1]
    u_test_wjac.append(u_wjac)

u_train_wjac = torch.stack(u_train_wjac)
u_test_wjac = torch.stack(u_test_wjac)

plt.figure()
plt.scatter(u_train_wjac, u_train, label='train')
plt.plot(u_train_wjac, u_train_wjac, label='y=x', color='black', linestyle='--')
plt.xlabel('E_dft - logdetjac')
plt.ylabel('E_dft')
plt.legend()
plt.show(block=False)

n_hidden = 64
n_layers = 2
model = MLP([x_train.shape[1]] +  [n_hidden] * n_layers + [1])

mlp_hyperparams = {'n_iter': 500,
    'lr': 5e-4,
    'use_scheduler': False,
    'step_schedule': 100,
}

out = train_mlp(model, x_train, x_test, u_train, u_test, **mlp_hyperparams, 
              with_tqdm=True)

out['x_train'] = x_train
out['x_test'] = x_test
out['u_train'] = u_train
out['u_test'] = u_test

plt.figure()
plt.plot(out['losses'][0], label='train')
plt.plot(out['losses'][1], label='test')
plt.legend()
plt.show(block=False)

# save_pickle_file(_, "mlp_" + isomer)
f = get_project_path() + "tests/{:s}_mlp_dic_training.pkl".format(isomer)
save_pickle_file(out, f)