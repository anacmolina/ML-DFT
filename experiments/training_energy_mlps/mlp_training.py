import sys

import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from flonacomldft.data_utils import (
    get_path
)

from flonacomldft.models import MLP
from flonacomldft.data_utils import save_pickle_file

import sklearn.model_selection
import torch.optim as optim
import tqdm

args = sys.argv
path_md = get_path() + args[1] + '_lcao_zmat.csv'
#path_nf = '~/ml_dft_project/nn_energy_training/first_mlps_models/u_nf_test.csv'
path_nf = 'x_nf_' + args[1] + '.csv'

df_md = pd.read_csv(path_md)
df_nf = pd.read_csv(path_nf)

zmat_md = torch.tensor(df_md.to_numpy()).float()[:4000, :]
zmat_nf = torch.tensor(df_nf.to_numpy()).float()[:1000, :]

zmat = torch.concat((zmat_md, zmat_nf)).detach().float()
#zmat = zmat_nf

sk_seed = 42
train_size = 0.8

y = zmat[:,-1]
y = np.round(y, decimals=4)
x = zmat[:,:-1]

x_mean = x.mean(0)
x_centered = x - x.mean(0)

x_centered_std = x_centered.std(0)
x_centered = x_centered / x_centered.std(0)

y_mean = y.mean()
y_centered = y - y.mean()

y_centered_std = y_centered.std()
y_centered = y_centered / y_centered.std()

arrays = [x_centered, y_centered]

data_split = sklearn.model_selection.train_test_split(*arrays, test_size=None,
                                                      train_size=train_size,
                                                      random_state=sk_seed,
                                                      shuffle=True,
                                                      stratify=None)

x_train, x_test, y_train, y_test = data_split

epochs = 20000
model = MLP([x.shape[1], 150, 150, 1])
opt = optim.Adam(model.parameters(), lr=0.00001)

def mse_loss(x,y):
    return ((model(x) - y[:,None]) ** 2).mean()

losses = []
losses_val = []
pbar =  tqdm.tqdm(range(epochs))
for i in pbar:
    opt.zero_grad()
    loss = mse_loss(x_train, y_train)
    loss.backward()
    opt.step()
    
    losses.append(loss.item())
    losses_val.append(mse_loss(x_test, y_test).item())
    pbar.set_description(f'Loss: {losses[-1]:.4f}')

plt.figure()
plt.plot(losses, label='train')
plt.plot(losses_val, label='test')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('loss_' + args[1] + '.jpg')
#plt.show()

plt.figure()
plt.plot(losses, label='train')
plt.plot(losses_val, label='test')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.yscale('log')
plt.savefig('logloss_' + args[1] + '.jpg')
#plt.show()

fig, ax = plt.subplots()
plt.scatter(model(x_train).detach().numpy(), y_train.detach().numpy(), label='train')
plt.scatter(model(x_test).detach().numpy(), y_test.detach().numpy(), label='test')
min_ = y_centered.min()
max_ = y_centered.max()
range_ = np.linspace(min_, max_, 2)
plt.plot(range_, range_, 'k')

plt.xlabel('Predicted')
plt.ylabel('Actual')
ax.set_aspect('equal')
plt.legend()
plt.savefig('correlation_' + args[1] + '.jpg')
#plt.show()

mlp_ = {
    'mlp_model': model,
    'x_mean': x_mean,
    'x_centered_std': x_centered_std,
    'y_mean': y_mean,
    'y_centered_std': y_centered_std
}

mlp_info = {
    'dataset': zmat,
    'dataset_split': data_split,
    'losses_train': losses,
    'losses_test': losses_val
}

save_pickle_file(mlp_, 'mlp_'+ args[1])
save_pickle_file(mlp_info, 'mlp_' + args[1] + '_info')
