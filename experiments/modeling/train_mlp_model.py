import torch

from flonacomldft.utils.io_utils import load_csv_file, save_pickle_file
from flonacomldft.internal_coordinates import Coordinates_mapping, join_data
from flonacomldft.models.mlp import MLP
from flonacomldft.train_mlp_from_data import train_mlp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.manual_seed(100)

# load data

mode_label = 1 #1
dataset_labels = ['md', 'flow']

zmat_train = torch.cat([load_csv_file("datasets/is{:d}_{:s}_train.csv".format(mode_label, dataset_label)) for dataset_label in dataset_labels])
zmat_test = torch.cat([load_csv_file("datasets/is{:d}_{:s}_test.csv".format(mode_label, dataset_label)) for dataset_label in dataset_labels])

# real centered frame

coord_mapping = Coordinates_mapping()
xs_train, logdetjacs_train, energies_train = coord_mapping.get_real_centered_from_internal(
                                    zmat_train[:, :12],
                                    zmat_train[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_train[:, 12]
                                    )

xs_test, logdetjacs_test, energies_test = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 12]
                                    )


n_hidden = 256
n_layers = 16
model = MLP([xs_train.shape[1]] +  [n_hidden] * n_layers + [1])

mlp_hyperparams = {'n_iter': 1000,
    'lr': 1e-4,
    'use_scheduler': False,
    'step_schedule': 100,
}

train = join_data(xs_train,
                energies_train,
                zmat_train[:, 14],
                logdetjacs_train,
                )

test = join_data(xs_test,
                energies_test,
                zmat_test[:, 14],
                logdetjacs_test)

out = train_mlp(model, train, test, **mlp_hyperparams, 
              with_tqdm=True)

import matplotlib.pyplot as plt
from flonacomldft.utils.plots import (
    plot_losses,
    plot_correlation_target_and_predict_value,
)

plot_losses(out['losses'][0], out['losses'][1], log_yscale=True)
plt.show()

plot_correlation_target_and_predict_value(
    energies_train,
    out['model'](xs_train.float()),
    energies_test,
    out['model'](xs_test.float()),
    title='MLP mode {:d}'.format(mode_label)
)
plt.show()


#f = "models/is{:d}_mlp_dic_training.pkl".format(mode_label)
#save_pickle_file(out, f)