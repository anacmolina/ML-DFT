### Import modules
import argparse
import time
import torch

from flonacomldft.utils.io_utils import (
    load_csv_file, 
    save_pickle_file,
    get_project_path,
    save_json_args 
)

from flonacomldft.internal_coordinates import (
    Coordinates_mapping,
    join_data
)

from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow
#from flonacomldft.parallel import set_seed
from flonacomldft.utils.io_utils import get_process_id


### Set equal seed for all ranks for parallel computations
num_seed = torch.randint(100, (1,)).item() #set_seed()

### Get starting time and process id
date_start = time.strftime('%Y-%m-%d %H:%M:%S')
process_id = get_process_id(date_start)

print('seed: ', num_seed)

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-np', '--num-procs', type=int, default=1)
parser.add_argument('-ni', '--n-iter', type=int, default=10)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-hdm', '--hidden-dim', type=int, default=64)
parser.add_argument('-hdp', '--hidden-depth', type=int, default=3)
parser.add_argument('-rs', '--random-seed', type=str, default=str(num_seed))
parser.add_argument('-path', '--folder-path', type=str, default='database/berendsen/datasets/')
parser.add_argument('-pid', '--process-id', type=int, default=int(process_id))

args = parser.parse_args()
args.date_start = date_start

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Set number of threads
torch.set_num_threads(int(args.num_procs))
print('n_thread_set: ', args.num_procs, torch.get_num_threads())

### Load datasets
mode_label = args.mode_label
zmat_train = load_csv_file(args.folder_path + "is{:d}_flow_train.csv".format(mode_label))
zmat_test = load_csv_file(args.folder_path + "is{:d}_flow_test.csv".format(mode_label))

### Get real center coordinates
coord_mapping = Coordinates_mapping()
xs_train, logdetjacs_train, energies_train = coord_mapping.get_real_centered_from_internal(
                                    zmat_train[:, :12],
                                    zmat_train[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_train[:, 12]
                                    )
xs_train = torch.tensor(xs_train, dtype=torch.float32)

xs_test, logdetjacs_test, energies_test = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 12]
                                    )
xs_test = torch.tensor(xs_test, dtype=torch.float32)

### Join real center coordinates, energies and logdetjacs
train = join_data(xs_train, energies_train, zmat_train[:, 13], logdetjacs_train).detach()
test = join_data(xs_test, energies_test, zmat_test[:, 13], logdetjacs_test).detach()

### Initialize Flow model
cov = torch.cov(xs_train.T).detach()
model = RealNVP_MLP(12,
                    n_blocks=args.n_blocks,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    base_cov=cov,
                    hidden_dim=args.hidden_dim,
                    hidden_depth=args.hidden_depth,
                    device=device,
                    )

### Train Flow model
out = train_flow(
    model,
    train,
    test,
    n_iter=args.n_iter,
    lr=args.learning_rate,
    use_scheduler=True,
    step_schedule=int(args.n_iter/10),
    save_splits=10,
    grad_clip=1e4,
    with_tqdm=False,
)

### Save end time
date_end = time.strftime('%Y-%m-%d %H:%M:%S')
args.date_end = date_end

args.algorithm = 'train_flow_model.py'

### Save arguments to output file
argparse_dict = vars(args)
out['args'] = argparse_dict

### Save output to pickle file
save_pickle_file(out, "is{:d}_flow_dic_training_{:d}.pkl".format(mode_label, args.process_id))

### Save arguments to json file
save_json_args(args, 'train_flow_model', args.process_id)

import numpy as np
import matplotlib.pyplot as plt
from flonacomldft.collective_variables import get_CVs
from flonacomldft.utils.plots import Flonaco_Plotter

xs = out['model'].sample(50)
zmats = coord_mapping.get_internal_from_real_centered(xs, isomer=0)[0]
cvs = np.array(get_CVs(zmats))

flow_plotter = Flonaco_Plotter()

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
flow_plotter.plot_losses(out['losses'], yscale=False, ax=axs[0])
flow_plotter.plot_collective_variables_on_time(cvs.T, ax=axs[1])
plt.savefig('is0_flow_training_{:d}.png'.format(args.process_id))

flow_plotter.plot_collective_variables_on_fes(cvs.T, label='Flow samples is{:d}'.format(mode_label))
plt.savefig('is0_flow_fes_{:d}.png'.format(args.process_id))