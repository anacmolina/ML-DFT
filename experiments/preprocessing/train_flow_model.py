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
from abflowmc.utils.io_utils import (
    get_path,
    set_str_date_to_int,
    load_csv_file,
    save_json_args,
    save_pickle_file
    )
# coordinates handling
from abflowmc.internal_coordinates import (
    Coordinates_mapping,
    )
from abflowmc.collective_variables import get_cvs_from_traj
from abflowmc.utils.data_processing import split_data_from_dataframe
# nf training
from abflowmc.models.real_nvp import RealNVP_MLP
from abflowmc.train_flow_from_data import train_flow

import matplotlib.pyplot as plt
from abflowmc.utils.plots import set_plot_sequential_data, plot_energy_surface

# set seed and date_start
num_seed = np.random.randint(0, 100)
date_start = set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))

print('Seed: ', num_seed)
print('Date_start: ', date_start)

# define arpase arguments
parser = argparse.ArgumentParser(description='Prepare experiment')
# execution params
parser.add_argument('-threads', '--threads', type=int, default=None)
parser.add_argument('-pid', '--process-id', type=int, default=date_start)
parser.add_argument('-rs', '--random-seed', type=int, default=num_seed)
parser.add_argument('-path', '--folder-path', type=str, default='andersen')
# training params
parser.add_argument('-isomer', '--isomer-label', type=int, default=0)
parser.add_argument('-etype', '--energy-type', type=str, default='dft')
parser.add_argument('-ni', '--n-iter', type=int, default=100)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
parser.add_argument('-bs', '--batch-size', type=int, default=100)
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-hdm', '--hidden-dim', type=int, default=64)
parser.add_argument('-hdp', '--hidden-depth', type=int, default=3)
parser.add_argument('-us', '--use-scheduler', type=bool, default=False)
parser.add_argument('-ss', '--step-scheduler', type=int, default=10)
parser.add_argument('-N', '--N', type=int, default=1000)

args = parser.parse_args()
args.date_start = date_start

print('Flow training')

# torch settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)
if args.threads is not None:
    torch.set_num_threads(args.threads)
    
torch.manual_seed(args.random_seed)

path_datasets = get_path() + '/' + args.folder_path + '/' + 'datasets'

# real centered coordinates datasets
flows_dataset = load_csv_file("is{:d}_flow_train.csv".format(args.isomer_label), path=path_datasets)[:args.N]
    
train, test = list(split_data_from_dataframe(flows_dataset, 0.8, 42))

print('Train size: ', train.shape[0])
print('Test size: ', test.shape[0])

cvs_test = test[:, 15:].detach().numpy()
    
# init flow model
dim = 12
cov = torch.cov(train[:, :dim].T).detach() + 1e-5 * torch.eye(train[:, :dim].shape[1]).detach()

model = RealNVP_MLP(dim=train[:, :dim].shape[1],
                    n_blocks=args.n_blocks,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    base_cov=cov,
                    hidden_dim=args.hidden_dim,
                    hidden_depth=args.hidden_depth,
                    device=device,
                    )


# path to save results
folder_to_save_results = 'results_flow_is{:d}_{:d}'.format(args.isomer_label, args.process_id)
path_to_save_results = os.getcwd() + '/' + folder_to_save_results

if not os.path.exists(path_to_save_results):
    os.makedirs(path_to_save_results)

path_to_save_results = os.getcwd() + '/' + folder_to_save_results + '/'

# training flow model
flow_dic = train_flow(
    model,
    train,
    test=test,
    n_iter=args.n_iter,
    lr=args.learning_rate,
    bs=args.batch_size,
    use_scheduler=args.use_scheduler,
    step_scheduler=args.step_scheduler,
    save_splits=1,
    grad_clip=1e4,
    with_tqdm=False,
    dim=dim,
)

# save end time
date_end = set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))
args.date_end = date_end

args.algorithm = 'train_flow_model'

argparse_dic = vars(args)
flow_dic['args'] = argparse_dic

# save output to pickle file
save_pickle_file(flow_dic, "is{:d}_flow_dic_{:d}.pkl".format(args.isomer_label, args.process_id), path=path_to_save_results)

# save arguments to json file
save_json_args(args, 'train_flow_model', args.process_id, path=path_to_save_results)

fig, axs = plt.subplots(1, 3, figsize=(30, 7))

train_losses = torch.tensor(flow_dic['train_losses']).detach()
test_losses = torch.tensor(flow_dic['test_losses']).detach()

axs[0].set_title("Isomer: {:d}, nb: {:d}, hdm: {:d}, hdp: {:d}".format(args.isomer_label, args.n_blocks, args.hidden_dim, args.hidden_depth))
set_plot_sequential_data(train_losses, avg=False, ax=axs[0], alpha=1, label='train', color='blue')
set_plot_sequential_data(test_losses, avg=False, ax=axs[0], alpha=1, label='test', color='orange')
axs[0].legend()
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Loss')

set_plot_sequential_data(torch.abs(train_losses), avg=False, ax=axs[1], alpha=1, label='train', color='blue')
set_plot_sequential_data(torch.abs(test_losses), avg=False, ax=axs[1], alpha=1, label='test', color='orange')
axs[1].set_yscale('log')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('|Loss|')
axs[1].legend()

coord_mapping = Coordinates_mapping(etype=args.energy_type)

xs = flow_dic['model'].sample(test.shape[0])
molecules = [coord_mapping.build_molecule_from_real_centered(x.reshape(1, -1), isomer=args.isomer_label)[0] for x in xs]
cvs = get_cvs_from_traj(molecules)

plot_energy_surface(fig=fig, ax=axs[2])
axs[2].scatter(cvs_test[:, 0], cvs_test[:, 1], s=25, label='test', color='orange')
axs[2].scatter(cvs[:, 0], cvs[:, 1], s=25, label='flow', color='cyan')
axs[2].legend()

plt.savefig(path_to_save_results + '/is{:d}_flow_training_{:d}.png'.format(args.isomer_label, args.process_id))