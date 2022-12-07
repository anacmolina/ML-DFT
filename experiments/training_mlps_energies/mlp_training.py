import torch
import pandas as pd
import numpy as np

from flonacomldft.utils.data_utils import get_path, save_pickle_file
from flonacomldft.models import center_values, MLP

from flonacomldft.train_mlp_from_data import train_mlp

import sklearn.model_selection

df_md = pd.read_csv(get_path() + 'is1_lcao_zmat.csv')
df_nf = pd.read_csv(get_path() + 'x_nf_is1.csv')

n_md = 3000
n_nf = 1500

zmat_md = torch.tensor(df_md.to_numpy()).float()[:n_md, :]
zmat_nf = torch.tensor(df_nf.to_numpy()).float()[:n_nf, :]
zmat = torch.concat((zmat_md, zmat_nf)).detach().float()

x = zmat[:, :-1].numpy()
y = zmat[:, -1].numpy()

sk_seed = 42
train_size = 0.8

arrays = [x, y]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(*arrays, test_size=None,
                                                      train_size=train_size,
                                                      random_state=sk_seed,
                                                      shuffle=True,
                                                      stratify=None)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

print(x_train.shape)

n_hidden = 5
n_layers = 5
model = MLP([x.shape[1], n_hidden, n_layers, 1])

_ = train_mlp(model, x_train, y_train, x_test, y_test, n_iter=5000, lr=1e-3)

save_pickle_file(_, "mlp_init_is1")