import warnings
warnings.filterwarnings('ignore')

# standard library imports
import os
import time
import argparse

# scientific library imports
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# flonaco library imports
from flonacomldft.utils.io_utils import (
    get_path,
    load_csv_file,
    load_pickle_file,
    save_csv_file,
    save_pickle_file,
    save_json_args,
    set_str_date_to_int
)

# coordinates handling
from flonacomldft.internal_coordinates import Coordinates_mapping
# normalizing flows
from flonacomldft.models.mixture import Mixture
# sampling methods
from flonacomldft.sampling import run_metropolis

# parallelization
import gpaw.mpi as mpi

# get rank size and set up communicator
ranks = np.arange(0, mpi.world.size)
rank = mpi.rank
comm = mpi.world.new_communicator(ranks)

print(f"Rank {rank} of {mpi.world.size} is running.")
mpi.world.barrier()

# set the random seed
num_seed = np.array([0])
date_start = np.array([0])

if rank==0:
    num_seed = np.random.randint(0, 100000, (1,))
    date_start = np.array([set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))])

comm.broadcast(num_seed, 0)
comm.broadcast(date_start, 0)

num_seed = num_seed[0]
date_start = date_start[0]

mpi.world.barrier()

print('Rank {} of {} has seed {}.'.format(rank, mpi.world.size, num_seed))
print('Rank {} of {} has date start {}.'.format(rank, mpi.world.size, date_start))

# def argparser arguments
parser = argparse.ArgumentParser(description='Multimodal sampling')
# execution arguments
parser.add_argument('-threads', '--threads', type=int, default=None, help='Number of threads')
parser.add_argument('-pid', '--process-id', type=int, default=date_start, help='Process ID')
parser.add_argument('-rs', '--random_seed', type=int, default=num_seed, help='Random seed')
parser.add_argument('-path', '--folder_path', type=str, default='andersen', help='Folder path')
# physical system arguments
parser.add_argument('-isomer', '--isomer-label', type=int, nargs='+', default=[0], help='Isomer label')
# sampling parameters
parser.add_argument('-nchains', '--n-chains', type=int, default=5, help='Number of chains')
parser.add_argument('-nsteps', '--n-steps', type=int, default=10, help='Number of steps')
parser.add_argument('-etype', '--energy-type', type=str, default='dft', help='Energy type')
parser.add_argument('-T', '--temperature', type=float, default=350, help='Temperature')

args = parser.parse_args()
args.date_start = date_start

# torch settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.threads is not None:
    torch.set_num_threads(args.threads)

print('Device: ', device)
print('Number of threads {} per rank {}'.format(torch.get_num_threads(), rank))

# set random seed
torch.manual_seed(args.random_seed)
mpi.world.barrier()

# isomer labels
isomer_labels = args.isomer_label

# mcmc parameters
n_chains = args.n_chains
n_steps = args.n_steps
energy_type = args.energy_type

# path to datasets
path_datasets = get_path() + '/' + args.folder_path + '/' + 'datasets'

# load datasets
coord_mapping = Coordinates_mapping()

def get_dataset(name, path, isomer_labels):
    
    zmats = [load_csv_file("is{:d}_{:s}.csv".format(isomer_label, name), path) for isomer_label in isomer_labels] 
    xs = [coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=isomer_label,
                                    energies=zmat_test[:, 12]
                                    ) for isomer_label, zmat_test in zip(isomer_labels, zmats)]

    xs = [torch.cat((x[0], x[2].reshape(-1, 1), 
                                 zmat[:, 13].reshape(-1, 1), 
                                 #x[1].reshape(-1, 1),
                                 ), dim=1) for x, zmat in zip(xs, zmats)]
    
    #xs = xs.flatten(start_dim=0, end_dim=1).to(torch.float32)

    return xs, zmats

dataset_labels = ['flow_train', 'flow_test']

flow_xs_test, flow_zmats_test = get_dataset(dataset_labels[1], path_datasets, isomer_labels)

for i in range(len(isomer_labels)):
    print('Isomer {:d} has {:d} samples'.format(isomer_labels[i], flow_xs_test[i].shape[0]))

# load models

# path to models

if 'mlp' in args.energy_type:
        add_mlp = '_mlp'
else:
    add_mlp = ''

path_models = get_path() + '/' + args.folder_path + '/' + 'models'

flow_models = [load_pickle_file("dict_flow_model_is{:d}{:s}.pkl".format(
                                isomer_labels[i], add_mlp), 
                                path=path_models)['model'] for i in range(len(isomer_labels)) ]
# initizalize mcmc chains
xs_init = torch.cat(flow_xs_test).clone()
xs_init = xs_init[torch.randperm(xs_init.shape[0])]
xs_init = xs_init[:n_chains]

print('xs_init shape: ', xs_init.shape)

if len(isomer_labels) == 1:
    mixture = False
    simulation_name = 'is{:d}'.format(isomer_labels[0])
else:
    mixture = True
    simulation_name = 'mixture'
    mixture_model = Mixture(flow_models, torch.tensor([0.5, 0.5]).detach())

if "mlp" in energy_type: 
    path_models = get_path() + '/' + args.folder_path + '/' + 'models'

    mlps_dic = [load_pickle_file("mlp_model_is{:d}.pkl".format(
                                isomer_labels[i]), 
                                path=path_models) 
                                for i in range(len(isomer_labels))]
else:
    mlps_dic = None

folder_to_save_results = 'results_multimodal_sampling_{:s}_{:d}'.format(simulation_name, args.process_id)
path_to_save_results = os.getcwd() + '/' + folder_to_save_results

if rank == 0:
    if not os.path.exists(folder_to_save_results):
        os.makedirs(folder_to_save_results)
        print('Folder created: ', folder_to_save_results)

mpi.world.barrier()



mh = run_metropolis(
    model=mixture_model,
    init=xs_init,
    n_chains=n_chains,
    n_steps=n_steps,
    id_run=0,
    energy_type=energy_type,
    mixture=mixture,
    T=args.temperature,
    frac_dft=0.0,
    with_tqdm=False,
    return_ratio=False,
    return_proposals=False,
    dft_folder_name=path_to_save_results + '/' + 'DFTComputations_{:d}'.format(args.process_id),
    scheduler=5,
    update_weigth=True,
    alpha = 0.5,
    mlp_models=mlps_dic,
)

mpi.world.barrier()

date_end = np.array([0])

# save end time
if rank == 0:
    date_end = np.array([set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))])

comm.broadcast(date_end, 0)
date_end = date_end[0]

mpi.world.barrier()
args.date_end = str(date_end)

args.algorithm = 'metropolis.py'

argparse_dic = vars(args)
mh['args'] = argparse_dic

# save output to pickle file
save_pickle_file(mh, "{:s}_mcmc_dic_{:d}.pkl".format(simulation_name, args.process_id), path = path_to_save_results)

# save output to json file
save_json_args(args, 'metropolis', args.process_id, path_to_save_results)

import matplotlib.pyplot as plt
from flonacomldft.utils.plots import set_plot_sequential_data 

accs = mh['accs']

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
set_plot_sequential_data(accs.mean(dim=1), avg=True, window_size=10, axis=1, ax=ax, label='acc_ratio')
ax.set_xlabel('Iteration')
ax.set_ylabel('Acceptance rate')
plt.savefig(path_to_save_results + '/acceptance_rate_{:d}.png'.format(args.process_id))
