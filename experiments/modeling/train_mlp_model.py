import torch

from flonacomldft.utils.io_utils import load_csv_file, save_pickle_file
from flonacomldft.internal_coordinates import Coordinates_mapping
from flonacomldft.models.mlp import MLP
from flonacomldft.train_mlp import train_mlp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.manual_seed(100)

# load data

mode_label = 1 #1

zmat_train = load_csv_file("datasets/is{:d}_md_train.csv".format(mode_label))
zmat_test = load_csv_file("datasets/is{:d}_md_test.csv".format(mode_label))

# real centered frame

coord_mapping = Coordinates_mapping()
xs_train, logdetjacs_train, energies_train = coord_mapping.get_real_centered_from_internal(
                                    zmat_train[:, :12],
                                    zmat_train[:, 12],
                                    isomer=mode_label,
                                    energies=zmat_train[:, 13]
                                    )

xs_test, logdetjacs_test, energies_test = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 12],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 13]
                                    )


n_hidden = 128
n_layers = 32
model = MLP([xs_train.shape[1]] +  [n_hidden] * n_layers + [1])

mlp_hyperparams = {'n_iter': 1500,
    'lr': 1e-4,
    'use_scheduler': False,
    'step_schedule': 100,
}

train = torch.cat((xs_train, logdetjacs_train.reshape(-1, 1), energies_train.reshape(-1, 1)), dim=1).to(torch.float32)
test = torch.cat((xs_test, logdetjacs_test.reshape(-1, 1), energies_test.reshape(-1, 1)), dim=1).to(torch.float32)

print(test)

out = train_mlp(model, train, test, mode_label, **mlp_hyperparams, 
              with_tqdm=True)

import matplotlib.pyplot as plt
from flonacomldft.utils.plots import (
    plot_losses,
    plot_correlation_target_and_predict_value
)

plot_losses(out['losses'][0], out['losses'][1])
plt.show()

plot_correlation_target_and_predict_value(
    energies_train,
    out['model'](xs_train.float()),
    energies_test,
    out['model'](xs_test.float()),
    title='MLP mode {:d}'.format(mode_label)
)
plt.show()

f = "models/is{:d}_mlp_dic_training.pkl".format(mode_label)
save_pickle_file(out, f)