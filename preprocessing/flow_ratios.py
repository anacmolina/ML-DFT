### Import modules

import os
import argparse
import time
import numpy as np

import torch

from flonacomldft.parallel import set_seed
from flonacomldft.utils.io_utils import get_process_id
from flonacomldft.utils.io_utils import (
    load_pickle_file,
    load_csv_file,
    save_pickle_file, 
    save_json_args
)

from flonacomldft.internal_coordinates import Coordinates_mapping
from flonacomldft.utils.diagnostics import get_acceptance_ratio, get_participation_ratio

import gpaw.mpi as mpi

### Set seed
num_seed = set_seed()
torch.manual_seed(num_seed)

mpi.world.barrier()

### Get start time and process id
date_start = time.strftime('%Y-%m-%d %H:%M:%S')
process_id = get_process_id(date_start)

print('seed: ', num_seed)

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-np', '--num-procs', type=int, default=1)
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-nc', '--n-chains', type=int, default=5)
parser.add_argument('-ns', '--n-steps', type=int, default=10)
parser.add_argument('-etype', '--energy-type', type=str, default='dft')
parser.add_argument('-flow', '--flow-file', type=str, default=None)
parser.add_argument('-mlp', '--mlp-file', type=str, default=None)
parser.add_argument('-rs', '--random-seed', type=str, default=str(num_seed))
parser.add_argument('-path', '--folder-path', type=str, default='database/berendsen/datasets/')
parser.add_argument('-pid', '--process-id', type=int, default=int(process_id))


args = parser.parse_args()
args.date_start = date_start

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Set number of threads
# torch.set_num_threads(args.num_procs)
# print('n_thread_set: ', args.num_procs)

### Set parameters
mode_label = args.mode_label
n_chains = args.n_chains
n_steps = args.n_steps
energy_type = args.energy_type

### Load data - real centered coordinates
coord_mapping = Coordinates_mapping()

zmat_test = load_csv_file(args.folder_path + "is{:d}_flow_test.csv".format(mode_label))

xs = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 12]
                                    )

xs = torch.cat((xs[0], xs[2].reshape(-1, 1), zmat_test[:, 13].reshape(-1, 1), xs[1].reshape(-1, 1)), dim=1)
xs = xs.to(torch.float32)

### Configs to initialize the chains

xs = xs[torch.randperm(xs.size()[0])]
xs = xs[:n_chains]

### MLP model
if args.mlp_file is not None and energy_type == 'mlp':
    mlp_model = load_pickle_file(args.mlp_file, path = os.getcwd() + '/')['model']
else:
    mlp_model = None

### Flow model
flow_models = load_pickle_file(args.flow_file, path = os.getcwd() + '/')['models'][::2]

### Run MH simulation and compute acceptance ratio

acceptance_ratios = torch.stack([get_acceptance_ratio(xs, flow_model, n_chains, n_steps, id_run, energy_type, mlp_model, return_ratios=True) for id_run, flow_model in enumerate(flow_models)])

### Compute participation ratio
from flonacomldft.utils.diagnostics import Target_Log_Prob

target_log_prob = Target_Log_Prob(energy_type=args.energy_type, mode_label=mode_label, mlp_model=mlp_model).target_log_prob
participation_ratios = torch.stack([get_participation_ratio(flow_model, target_log_prob, n_prop=n_chains*n_steps) for flow_model in flow_models])

### Save results

date_end = time.strftime('%Y-%m-%d %H:%M:%S')
args.date_end = date_end

args.algorithm = 'compute_ratios.py'

save_json_args(args, 'compute_ratios', args.process_id, os.getcwd() + '/')

f = "is{:d}_mh_ratios_{:d}.pkl".format(mode_label, args.process_id)

out = {'acc_ratios': acceptance_ratios, 
       'part_ratios': participation_ratios}

save_pickle_file(out, f, path = os.getcwd() + '/')

import matplotlib.pyplot as plt

print(acceptance_ratios)
print(participation_ratios)

def plot_acceptance_ratio_nsteps(acceptance_ratios, split=10, alpha=0.2, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for i, acceptance_ratio in enumerate(acceptance_ratios):
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(acceptance_ratio, color=color, alpha=alpha)#, label='Model {:d}'.format(i))
        acc_ratio_avg = np.lib.stride_tricks.sliding_window_view(acceptance_ratio, split).mean(axis=1)
        ax.plot(acc_ratio_avg, color=color, label='Model {:d} - average'.format(i))
    ax.set_xlabel('n_steps')
    ax.set_ylabel('acceptance ratio')
    ax.legend()

def plot_participation_ratio(participation_ratios, marker='o-', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(participation_ratios, marker, label='participation ratio')
    ax.set_xlabel('flow model during training')
    ax.set_ylabel('participation ratio')
    ax.legend()

def plot_acceptance_ratio_models(acceptance_ratios, marker='o-', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(acceptance_ratios, marker, label='acceptance ratio')
    ax.set_xlabel('flow model during training')
    ax.set_ylabel('acceptance ratio')
    ax.legend()


fig = plt.figure(figsize=(10, 8))

ax1 = plt.subplot(222)
plot_acceptance_ratio_models(acceptance_ratios.detach().numpy()[:, -1], 'o-', ax=ax1)

ax2 = plt.subplot(221)
plot_participation_ratio(participation_ratios.detach().numpy(), ax=ax2)

ax3 = plt.subplot(212)
plot_acceptance_ratio_nsteps(acceptance_ratios.detach().numpy()[8:], ax=ax3)

plt.savefig('acceptance_participation_ratios_{:d}.png'.format(args.process_id))