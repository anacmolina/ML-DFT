import torch

from flonacomldft.utils.io_utils import load_csv_file

from flonacomldft.models.mlp import MLP
from flonacomldft.train_mlp_from_data import train_mlp

from flonacomldft.utils.io_utils import save_pickle_file
import matplotlib.pyplot as plt

mode_label = 1 #or 2

xs_md_train = load_csv_file("datasets/is{:d}_md_train.csv".format(mode_label))
xs_flow_train = load_csv_file("datasets/is{:d}_flow_test.csv".format(mode_label))

xs_md_test = load_csv_file("datasets/is{:d}_md_test.csv".format(mode_label))
xs_flow_test = load_csv_file("datasets/is{:d}_flow_test.csv".format(mode_label))

# concat of MD configs and random generated samples from flows
x_train = torch.cat((xs_md_train, xs_flow_train))
x_test = torch.cat((xs_md_test, xs_flow_test))

# remove logdetcat and split input (internal coordinates) and output (energies)
u_train = x_train.clone()[:, 13]
u_test = x_test.clone()[:, 13]
x_train = x_train.clone()[:, :12]
x_test = x_test.clone()[:, :12]

n_hidden = 64
n_layers = 2
model = MLP([x_train.shape[1]] +  [n_hidden] * n_layers + [1])

mlp_hyperparams = {'n_iter': 5000,
    'lr': 5e-4,
    'use_scheduler': False,
    'step_schedule': 100,
}

out = train_mlp(model, x_train, x_test, u_train, u_test, **mlp_hyperparams, 
              with_tqdm=True)

import matplotlib.pyplot as plt
from flonacomldft.utils.plots import (
    plot_losses,
    plot_correlation_target_and_predict_value
)

plot_losses(out['losses'][0], out['losses'][1])
plt.show(block=False)

plot_correlation_target_and_predict_value(
    u_train,
    out['model'].predict(x_train),
    u_test,
    out['model'].predict(x_test),
    title='MLP mode {:d}'.format(mode_label)
)
plt.show(block=False)

f = "is{:d}_mlp_dic_training.pkl".format(mode_label)
save_pickle_file(out, f)