### Import modules
import argparse

import time
import torch

from flonacomldft.utils.io_utils import (
    load_csv_file, 
    save_pickle_file,
    save_json_args, 
    get_process_id,
)

from flonacomldft.internal_coordinates import (
    Coordinates_mapping,
    join_data
)

from flonacomldft.models.mlp import MLP
from flonacomldft.train_mlp_from_data import train_mlp
#from flonacomldft.parallel import set_seed

### Set random seed
num_seed = torch.randint(100, (1,)).item() #set_seed()

### Set starting time and process id
date_start = time.strftime('%Y-%m-%d %H:%M:%S')
process_id = get_process_id(date_start)

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
parser.add_argument('-path', '--folder-path', type=str, default='berendsen/datasets/')
parser.add_argument('-pid', '--process-id', type=int, default=process_id)
args = parser.parse_args()

args.date_start = date_start

### Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Set number of threads and random seed to PyTorch
torch.set_num_threads(int(args.num_procs))
torch.manual_seed(num_seed)

### Load data
mode_label = args.mode_label

zmat_train = load_csv_file(args.folder_path + "is{:d}_mlp_train.csv".format(mode_label))
zmat_test = load_csv_file(args.folder_path + "is{:d}_mlp_test.csv".format(mode_label))

### Get real centered coordinates

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

### MLP hyperparameters
n_hidden = args.hidden_dim
n_layers = args.hidden_depth
model = MLP([xs_train.shape[1]] +  [n_hidden] * n_layers + [1])

print('threads: {:d}'.format(torch.get_num_threads()))

train = join_data(xs_train,
                energies_train,
                zmat_train[:, 13],
                logdetjacs_train,
                ).detach()

test = join_data(xs_test,
                energies_test,
                zmat_test[:, 13],
                logdetjacs_test).detach()

out = train_mlp(model, 
                train, 
                test, 
                n_iter=args.n_iter,
                lr=args.learning_rate,
                use_scheduler=True,
                step_schedule=1000,
                with_tqdm=False)

date_end = time.strftime('%Y-%m-%d %H:%M:%S')
args.date_end = date_end

### Add arguments to output dictionary
argparse_dict = vars(args)
out['args'] = argparse_dict

### Save output dictionary
f = "is{:d}_mlp_dic_training_{:d}.pkl".format(mode_label, args.process_id)
save_pickle_file(out, f)

### Save arguments
save_json_args(args, 'train_mlp_model', args.process_id)

### Plot training results
import matplotlib.pyplot as plt
from flonacomldft.utils.plots import plot_mlp_training

plot_training_results = plot_mlp_training(out)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].set_title('Mode {:d}'.format(mode_label))
plot_training_results.plot_loss(ax=axs[0])
plot_training_results.plot_correlation([train, test], ax=axs[1])
plt.savefig('is{:d}_mlp_training_{:d}.png'.format(mode_label, args.process_id))