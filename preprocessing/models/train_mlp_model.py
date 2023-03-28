import argparse

import time
import torch

from flonacomldft.utils.io_utils import (
    load_csv_file, 
    save_pickle_file,
    save_json_args, 
    get_date_process_id
)

from flonacomldft.internal_coordinates import (
    Coordinates_mapping,
    join_data
)

from flonacomldft.models.mlp import MLP
from flonacomldft.train_mlp_from_data import train_mlp
from flonacomldft.parallel import set_seed

num_seed = set_seed()

# for naming files
date_start, process_id = get_date_process_id()

print('seed: ', num_seed)

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-np', '--num-procs', type=int, default=1)
parser.add_argument('-ni', '--n-iter', type=int, default=10)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-hdm', '--hidden-dim', type=int, default=64)
parser.add_argument('-hdp', '--hidden-depth', type=int, default=3)
parser.add_argument('-rs', '--random-seed', type=str, default=str(num_seed))
parser.add_argument('-path', '--folder-path', type=str, default='database/')
parser.add_argument('-pid', '--process-id', type=id, default=process_id)
args = parser.parse_args()

args.date_start = date_start
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(int(args.num_procs))
torch.manual_seed(num_seed)

# load data

mode_label = args.mode_label
zmat_train = load_csv_file(args.folder_path + "is{:d}_md_train.csv".format(mode_label))
zmat_test = load_csv_file(args.folder_path + "is{:d}_md_test.csv".format(mode_label))

#dataset_labels = ['md', 'flow']

#zmat_train = torch.cat([load_csv_file("database/datasets/is{:d}_{:s}_train.csv".format(mode_label, dataset_label)) for dataset_label in dataset_labels])
#zmat_test = torch.cat([load_csv_file("database/datasets/is{:d}_{:s}_test.csv".format(mode_label, dataset_label)) for dataset_label in dataset_labels])


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

f = args.folder_path + "is{:d}_mlp_dic_training_{:d}.pkl".format(mode_label, args.process_id)
save_pickle_file(out, f)

save_json_args(args, 'train_mlp_model', process_id)

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