import torch
import pandas as pd

from flonacomldft.utils.data_utils import (
    get_path, 
    save_pickle_file, 
    split_data_from_csv
)

from flonacomldft.models import MLP

from flonacomldft.train_mlp_from_data import train_mlp

#import sklearn.model_selection
#
#def split_data_from_csv(df, train_size, sk_seed):
#    zmat = torch.tensor(df.to_numpy()).float()
#    x = zmat[:, :-1].numpy()
#    y = zmat[:, -1].numpy()
#
#    arrays = [x, y]
#
#    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(*arrays, test_size=None,
#                                                      train_size=train_size,
#                                                      random_state=sk_seed,
#                                                      shuffle=True,
#                                                      stratify=None)
#
#    x_train = torch.from_numpy(x_train).float()
#    y_train = torch.from_numpy(y_train).float()
#    x_test = torch.from_numpy(x_test).float()
#    y_test = torch.from_numpy(y_test).float()
#
#    return x_train, x_test, y_train, y_test 

sk_seed = 42
train_size = 0.8

n_md = 2500 # 5000 steps in total
df_md = pd.read_csv(get_path() + 'is1_lcao_zmat.csv').loc[:n_md]

x_train_md, x_test_md, y_train_md, y_test_md = split_data_from_csv(df_md, train_size, sk_seed)

n_nf = 1500 # 2500 configs in total
df_nf = pd.read_csv(get_path() + 'x_nf_is1.csv').loc[:n_nf]

x_train_nf, x_test_nf, y_train_nf, y_test_nf = split_data_from_csv(df_nf, train_size, sk_seed)

x_train = torch.cat((x_train_md, x_train_nf))
y_train = torch.cat((y_train_md, y_train_nf))
x_test = torch.cat((x_test_md, x_test_nf))
y_test = torch.cat((y_test_md, y_test_nf))

n_hidden = 50
n_layers = 50
model = MLP([x_train.shape[1], n_hidden, n_layers, 1])

mlp_hyperparams = {'n_iter': 1000,
    'lr': 5e-3,
    'use_scheduler': False,
    'step_schedule': 100,
    'grad_clip': 1e4,
}

_ = train_mlp(model, x_train, y_train, x_test, y_test, **mlp_hyperparams)

save_pickle_file(_, "mlp_is1")