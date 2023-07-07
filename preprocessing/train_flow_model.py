import warnings
warnings.filterwarnings("ignore")

# standard library imports
import os
import time
import argparse

# scientific library imports
import torch
import numpy as np

# flonaco imports
# io handling
from flonacomldft.utils.io_utils import (
    get_path,
    set_str_date_to_int,
    load_csv_file,
    save_json_args,
    save_pickle_file
    )
# coordinates handling
from flonacomldft.internal_coordinates import (
    Coordinates_mapping,
    join_data,
    )
# nf training
from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow

# set seed and date_start
num_seed = np.random.randint(0, 100)
date_start = set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))

print('seed: ', num_seed)
print('date_start: ', date_start)

# define arpase arguments
parser = argparse.ArgumentParser(description='Prepare experiment')
# execution params
parser.add_argument('-threads', '--threads', type=int, default=1)
parser.add_argument('-pid', '--process-id', type=int, default=date_start)
parser.add_argument('-rs', '--random-seed', type=int, default=num_seed)
parser.add_argument('-path', '--folder-path', type=str, default='emt_berendsen')
# training params
parser.add_argument('-isomer', '--isomer-label', type=int, default=0)
parser.add_argument('-ni', '--n-iter', type=int, default=1000)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
parser.add_argument('-bs', '--batch-size', type=int, default=128)
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-hdm', '--hidden-dim', type=int, default=64)
parser.add_argument('-hdp', '--hidden-depth', type=int, default=3)
parser.add_argument('-us', '--use-scheduler', type=bool, default=False)
# sampling params
parser.add_argument('-ratios', '--do-ratios', type=bool, default=False)
parser.add_argument('-etype', '--energy-type', type=str, default='emt')
parser.add_argument('-prop', '--n-prop', type=int, default=100)

args = parser.parse_args()
args.date_start = date_start

# torch settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(args.threads)
torch.manual_seed(args.random_seed)

path_datasets = get_path() + '/' + args.folder_path + '/' + 'datasets'

# internal coordinates datasets
zmat_train = load_csv_file("is{:d}_flow_train.csv".format(args.isomer_label), path=path_datasets)
zmat_test = load_csv_file("is{:d}_flow_test.csv".format(args.isomer_label), path=path_datasets)


# real center coordinates datasets
coord_mapping = Coordinates_mapping()
xs_train, logdetjacs_train, energies_train = coord_mapping.get_real_centered_from_internal(
                                    zmat_train[:, :12],
                                    zmat_train[:, 14],
                                    isomer=args.isomer_label,
                                    energies=zmat_train[:, 12]
                                    )
xs_train = torch.tensor(xs_train, dtype=torch.float32)

xs_test, logdetjacs_test, energies_test = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=args.isomer_label,
                                    energies=zmat_test[:, 12]
                                    )
xs_test = torch.tensor(xs_test, dtype=torch.float32)

# join real center coordinates, energies and logdetjacs in a single tensor
train = join_data(xs_train, energies_train, zmat_train[:, xs_train.shape[1]+1], logdetjacs_train).detach()
test = join_data(xs_test, energies_test, zmat_test[:, xs_test.shape[1]+1], logdetjacs_test).detach()

# init flow model

cov = torch.cov(xs_train.T).detach() + 1e-5 * torch.eye(xs_train.shape[1]).detach()

model = RealNVP_MLP(dim=xs_train.shape[1],
                    n_blocks=args.n_blocks,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    base_cov=cov,
                    hidden_dim=args.hidden_dim,
                    hidden_depth=args.hidden_depth,
                    device=device,
                    )


# path to save results
folder_to_save_results = 'results_training_flow_is{:d}_{:d}'.format(args.isomer_label, args.process_id)
path_to_save_results = os.getcwd() + '/' + folder_to_save_results

if not os.path.exists(path_to_save_results):
    os.makedirs(path_to_save_results)

path_to_save_results = os.getcwd() + '/' + folder_to_save_results + '/'

# training flow model
flow_dic = train_flow(
    model,
    train,
    test,
    n_iter=args.n_iter,
    lr=args.learning_rate,
    batch_size=args.batch_size,
    use_scheduler=args.use_scheduler,
    step_schedule=args.n_iter/5,
    save_splits=10,
    grad_clip=1e4,
    with_tqdm=False,
    compute_part_ratio=args.do_ratios,
    energy_type=args.energy_type,
    n_prop=args.n_prop,
    path=path_to_save_results,
)

# save end time
date_end = set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))
args.date_end = date_end

args.algorithm = 'train_flow_model'

argparse_dic = vars(args)
flow_dic['args'] = argparse_dic

# save output to pickle file
save_pickle_file(flow_dic, "is{:d}_flow_dic_training_{:d}.pkl".format(args.isomer_label, args.process_id), path=path_to_save_results)

# save arguments to json file
save_json_args(args, 'train_flow_model', args.process_id, path=path_to_save_results)

# plotting libraries
import matplotlib.pyplot as plt
from flonacomldft.utils.plots import (
    Flonaco_Plotter,
)

from flonacomldft.utils.plots import set_plot_sequential_data as set_plot_iteration

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.set_title("ml: {:d}, nb: {:d}, hdm: {:d}, hdp: {:d}".format(args.isomer_label, args.n_blocks, args.hidden_dim, args.hidden_depth))
set_plot_iteration(flow_dic['losses'][0], avg=False, ax=ax, alpha=1, label='train')
set_plot_iteration(flow_dic['losses'][1], avg=False, ax=ax, alpha=1, label='test')
ax.legend()
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
plt.savefig(path_to_save_results + '/is{:d}_flow_losses_{:d}.png'.format(args.isomer_label, args.process_id))

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.set_title("ml: {:d}, nb: {:d}, hdm: {:d}, hdp: {:d}".format(args.isomer_label, args.n_blocks, args.hidden_dim, args.hidden_depth))
set_plot_iteration(torch.log(torch.abs(flow_dic['losses'][0])), avg=False, ax=ax, alpha=1, label='train')
set_plot_iteration(torch.log(torch.abs(flow_dic['losses'][1])), avg=False, ax=ax, alpha=1, label='test')
ax.legend()
ax.set_xlabel('Iterations')
ax.set_ylabel('$\log$(|loss|)')
plt.savefig(path_to_save_results + '/is{:d}_flow_log_losses_{:d}.png'.format(args.isomer_label, args.process_id))

if args.do_ratios:
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    set_plot_iteration(flow_dic['part_ratios'].detach().numpy(), avg=True, window_size=5, axis=1, ax=ax, label='part_ratio')
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Participation ratio')
    ax.legend()
    plt.savefig(path_to_save_results + '/is{:d}_flow_part_ratio_{:d}.png'.format(args.isomer_label, args.process_id))

from flonacomldft.collective_variables import get_CVs

xs = flow_dic['model'].sample(150)
zmats = coord_mapping.get_internal_from_real_centered(xs, isomer=args.isomer_label)[0]
cvs = np.array(get_CVs(zmats))

flow_plotter = Flonaco_Plotter()

flow_plotter.plot_collective_variables_on_fes(cvs.T, label='Flow samples is{:d}'.format(args.isomer_label))
plt.title("ml: {:d}, nb: {:d}, hdm: {:d}, hdp: {:d}".format(args.isomer_label, args.n_blocks, args.hidden_dim, args.hidden_depth))
plt.savefig(path_to_save_results + '/is{:d}_flow_fes_{:d}.png'.format(args.isomer_label, args.process_id))





