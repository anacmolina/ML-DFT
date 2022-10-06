import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from flonacomldft.data_utils import (
    get_path
)

from flonacomldft.models import MLP, Uncentered_MLP
from flonacomldft.data_utils import save_pickle_file
import sklearn.model_selection

from flonacomldft.train_mlp_from_data import train_mlp

isomer = 'is2'

path_md = get_path() + isomer + '_lcao_zmat.csv'
path_nf = get_path() + 'x_nf_' + isomer + '.csv'

df_md = pd.read_csv(path_md)
df_nf = pd.read_csv(path_nf)

zmat_md = torch.tensor(df_md.to_numpy()).float()
#zmat_nf = torch.tensor(df_nf.to_numpy()).float()[:200, :]

#zmat = torch.concat((zmat_md, zmat_nf)).detach().float()
zmat = zmat_md

y = zmat[:,-1]
y = np.round(y, decimals=4)
x = zmat[:,:-1]

n_hidden = 50
n_layers = 50
model = MLP([x.shape[1], n_hidden, n_layers, 1])

_ = train_mlp(model, x, y, n_iter=5000, lr=1e-3)

losses = _['mlp_info']['losses']
losses_val = _['mlp_info']['losses_test']

plt.figure()
plt.plot(losses, label='train')
plt.plot(losses_val, label='test')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
#plt.savefig('loss_' + args[1] + '.jpg')
plt.show()

plt.figure()
plt.plot(losses, label='train')
plt.plot(losses_val, label='test')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.yscale('log')
#plt.savefig('logloss_' + isomer + '.jpg')
plt.show()

model = Uncentered_MLP(_['mlp_model'])

sk_seed = 42
train_size = 0.8

arrays = [x, y]

data_split = sklearn.model_selection.train_test_split(
    *arrays,
    test_size=None,
    train_size=train_size,
    random_state=sk_seed,
    shuffle=True,
    stratify=None,
)

x_train, x_test, y_train, y_test = data_split

fig, ax = plt.subplots()
plt.scatter(model(x_train).detach().numpy(), y_train.detach().numpy(), label='train')
plt.scatter(model(x_test).detach().numpy(), y_test.detach().numpy(), label='test')
min_ = y.min()
max_ = y.max()
range_ = np.linspace(min_, max_, 2)
plt.plot(range_, range_, 'k')

plt.xlabel('Predicted')
plt.ylabel('Actual')
ax.set_aspect('equal')
plt.legend()
#plt.savefig('correlation_' + isomer + '.jpg')
plt.show()
 