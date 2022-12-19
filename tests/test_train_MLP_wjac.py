import torch
import pandas as pd

from flonacomldft.utils.data_utils import (
    get_path, 
    save_pickle_file, 
    split_data_from_dataframe
)

from flonacomldft.models.mlp import MLP

from flonacomldft.train_mlp_from_data import train_mlp

isomer = 'is1'

sk_seed = 42
train_size = 0.8

n_md = 2500 # 5000 steps in total
df_md = pd.read_csv(get_path() + isomer + '_lcao_zmat.csv').loc[:n_md]

x_train_md, x_test_md, u_train_md, u_test_md = split_data_from_dataframe(df_md, train_size, sk_seed)

n_nf = 1500 # 2500 configs in total
df_nf = pd.read_csv(get_path() + 'x_nf_'+ isomer +'.csv').loc[:n_nf]

x_train_nf, x_test_nf, u_train_nf, u_test_nf = split_data_from_dataframe(df_nf, train_size, sk_seed)

x_train = torch.cat((x_train_md, x_train_nf))
u_train = torch.cat((u_train_md, u_train_nf))
x_test = torch.cat((x_test_md, x_test_nf))
u_test = torch.cat((u_test_md, u_test_nf))

n_hidden = 50
n_layers = 5
model = MLP([x_train.shape[1], n_hidden, n_layers, 1])

mlp_hyperparams = {'n_iter': 10,
    'lr': 5e-3,
    'use_scheduler': False,
    'step_schedule': 100,
    'grad_clip': 1e4,
}

_ = train_mlp(model, x_train, x_test, u_train, u_test, **mlp_hyperparams)

# save_pickle_file(_, "mlp_" + isomer)