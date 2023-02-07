import torch

from flonacomldft.internal_coordinates import Coordinates_mapping
from flonacomldft.utils.io_utils import load_csv_file

from flonacomldft.models.mlp import MLP
from flonacomldft.train_mlp import train_mlp

from flonacomldft.utils.io_utils import save_pickle_file
import matplotlib.pyplot as plt

mode_label = 0 #or 1

xs_md_train = load_csv_file("datasets/is{:d}_md_train.csv".format(mode_label))
xs_flow_train = load_csv_file("datasets/is{:d}_flow_test.csv".format(mode_label))

xs_md_test = load_csv_file("datasets/is{:d}_md_test.csv".format(mode_label))
xs_flow_test = load_csv_file("datasets/is{:d}_flow_test.csv".format(mode_label))

# concat of MD configs and random generated samples from flows
train = torch.cat((xs_md_train, xs_flow_train))
test = torch.cat((xs_md_test, xs_flow_test))

# remove logdetcat and split input (internal coordinates) and output (energies)
u_train = train.clone()[:, 13]
u_test = test.clone()[:, 13]
x_train = train.clone()[:, :12]
x_test = test.clone()[:, :12]

n_hidden = 128
n_layers = 16
model = MLP([x_train.shape[1]] +  [n_hidden] * n_layers + [1])

mlp_hyperparams = {'n_iter': 1000,
    'lr': 5e-5,
    'use_scheduler': False,
    'step_schedule': 100,
}

out = train_mlp(model, train, test, mode_label, **mlp_hyperparams, 
              with_tqdm=True)

import matplotlib.pyplot as plt
from flonacomldft.utils.plots import (
    plot_losses,
    plot_correlation_target_and_predict_value
)

plot_losses(out['losses'][0], out['losses'][1])
plt.show()

zs_train, logdetjac_train, us_train = train[:, :12], train[:, 12], train[:, 13]
zs_test, logdetjac_test, us_test = test[:, :12], test[:, 12], test[:, 13] 

coord_mapping = Coordinates_mapping()

zs_train, logdetjac_train, us_train = coord_mapping.get_real_centered_from_internal(zs_train, logdetjac_train, isomer=mode_label-1, energies=us_train)
zs_test, logdetjac_test, us_test = coord_mapping.get_real_centered_from_internal(zs_test, logdetjac_test, isomer=mode_label-1, energies=us_test)

plot_correlation_target_and_predict_value(
    us_train,
    out['model'](zs_train.float()),
    us_test,
    out['model'](zs_test.float()),
    title='MLP mode {:d}'.format(mode_label)
)
plt.show()

#f = "is{:d}_mlp_dic_training.pkl".format(mode_label)
#save_pickle_file(out, f)