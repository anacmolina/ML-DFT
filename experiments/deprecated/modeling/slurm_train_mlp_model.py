import argparse
import time
import torch
import numpy as np

from flonacomldft.utils.io_utils import load_csv_file, save_pickle_file, get_path
from flonacomldft.internal_coordinates import Coordinates_mapping, join_data
from flonacomldft.models.mlp import MLP
from flonacomldft.train_mlp_from_data import train_mlp

# for naming files
date_start = time.strftime('%H:%M:%S %d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)

print(get_path())

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-np', '--num-procs', type=int, default=8)
parser.add_argument('-ni', '--n-iter', type=int, default=5000)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-hdm', '--hidden-dim', type=int, default=64)
parser.add_argument('-hdp', '--hidden-depth', type=int, default=3)
parser.add_argument('-id', '--slurm-id', type=str, default=str(random_id))

args = parser.parse_args()

args.date_start = date_start
args.random_seed = random_id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(int(args.num_procs))

torch.manual_seed(100)

# load data

mode_label = args.mode_label
#dataset_labels = ['md', 'flow']

#zmat_train = torch.cat([load_csv_file("database/datasets/is{:d}_{:s}_train.csv".format(mode_label, dataset_label)) for dataset_label in dataset_labels])
#zmat_test = torch.cat([load_csv_file("database/datasets/is{:d}_{:s}_test.csv".format(mode_label, dataset_label)) for dataset_label in dataset_labels])

zmat_train = load_csv_file("database/andersen/is{:d}_md_train_{:d}.csv".format(mode_label, 2206336))
zmat_test = load_csv_file("database/andersen/is{:d}_md_test_{:d}.csv".format(mode_label, 2206336))


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


n_hidden = args.hidden_dim
n_layers = args.hidden_depth
model = MLP([xs_train.shape[1]] +  [n_hidden] * n_layers + [1])

print(args)
print('threads: {:d}'.format(torch.get_num_threads()))

mlp_hyperparams = {'n_iter': args.n_iter,
    'lr': args.learning_rate,
    'use_scheduler': False,
    'step_schedule': 100,
}

train = join_data(xs_train,
                energies_train,
                zmat_train[:, 14],
                logdetjacs_train,
                ).detach()

test = join_data(xs_test,
                energies_test,
                zmat_test[:, 14],
                logdetjacs_test).detach()

out = train_mlp(model, train, test, **mlp_hyperparams, 
              with_tqdm=True)

date_end = time.strftime('%H:%M:%S %d-%m-%Y')
args.date_end = date_end

argparse_dict = vars(args)
out['args'] = argparse_dict

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

#TODO: save hyperparams to a JSON file
#TODO: test how many core are good for just one training
#TODO: where to save the .log file
#TODO: how to save the files
#TODO: write a tasks, uses a small grid spacing

#f = "models/mlp_tracking/is{:d}_mlp_dic_training_{:s}.pkl".format(mode_label, args.slurm_id)
#save_pickle_file(out, f)

#import json

#filename_args = "args_" + args.slurm_id + ".json"

#with open(get_path() + "models/mlp_tracking/" + filename_args, "w") as outfile:
#    json.dump(argparse_dict, outfile)