import warnings
warnings.filterwarnings('ignore')

# standar libraries
import os
import time
import argparse

# scientific libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# flonaco functions
from flonacomldft.utils.io_utils import (
    get_project_path,
    load_csv_file,
    load_pickle_file,
    save_csv_file,
    save_pickle_file,
    save_json_args,
    set_str_date_to_int
)

# coordinates handling
from flonacomldft.internal_coordinates import Coordinates_mapping
from flonacomldft.utils.data_processing import split_data_from_dataframe
# models building
from flonacomldft.models.mixture import Mixture
# sampling methods
from flonacomldft.sampling import run_metropolis
# collective variables
from flonacomldft.internal_coordinates import get_collective_variables_from_xs
# plots
import matplotlib.pyplot as plt
from flonacomldft.utils.plots import set_plot_sequential_data

# parallelization
import gpaw.mpi as mpi
from ase.parallel import parprint as print

# get rank size and set up communicator
ranks = np.arange(0, mpi.world.size)
rank = mpi.rank
comm = mpi.world.new_communicator(ranks)

print(f"Rank {rank} of {mpi.world.size} is running.")
mpi.world.barrier()

# share random seed
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
print('Rank {} of {} has date {}.'.format(rank, mpi.world.size, date_start))

parser = argparse.ArgumentParser(description='Multimodal sampling of the potential energy surface.')
# execution parameters
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
parser.add_argument('-ftype', '--flow_type', type=str, default=None)
parser.add_argument('-upw', '--update-weights', type=bool, default=True)
parser.add_argument('-schw', '--scheduler-weights', type=int, default=10)
parser.add_argument('-alpha', '--alpha', type=float, default=0.5)
parser.add_argument('-frac', '--frac-computed', type=float, default=0.2)
parser.add_argument('-npsID', '--mlps-id', type=int, nargs='+', default=None, help='Neural predictors ID')
parser.add_argument('-nfsID', '--flows-id', type=int, nargs='+', default=None, help='Number of neural predictors')
parser.add_argument('-savepts', '--checkpoints', type=int, default=None, help='Checkpoints scheduler')
parser.add_argument('-slice', '--slice', type=int, default=30, help='Slice of the dataset')
parser.add_argument('-trainmlps', '--train-mlp-models', type=bool, default=False, help='Train MLP models')
parser.add_argument('-tmlpss', '--train-mlp-scheduler', type=int, default=None, help='Train flow models')
parser.add_argument('-lr', '--mlp-learning-rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('-bs', '--mlp-batch-size', type=int, default=500, help='Batch size')
parser.add_argument('-niter', '--mlp-n-iter', type=int, default=2500, help='Number of iterations')
parser.add_argument('-us', '--mlp-use-scheduler', type=bool, default=False, help='Use scheduler')
parser.add_argument('-ss', '--mlp-step-scheduler', type=int, default=100, help='Step scheduler')

# parse arguments
args = parser.parse_args()
args.date_start = date_start

# torch settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.threads is not None:
    torch.set_num_threads(args.threads)

print('Device: ', device)

# set random seed
torch.manual_seed(int(args.random_seed))
mpi.world.barrier()

# isomer labels
isomer_labels = args.isomer_label
dim=12

# mcmc parameters
n_chains = args.n_chains
n_steps = args.n_steps
energy_type = args.energy_type

# path to datasets
path_datasets = get_project_path() + '/database/' + args.folder_path + '/' + 'datasets'

# load datasets

flows_dataset = [load_csv_file('is{:d}_flow_train.csv'.format(isomer_labels[i] 
                                                                ), 
                                 path_datasets) for i in range(len(isomer_labels))]

flows_train = []
flows_test = []

for i in range(len(isomer_labels)):
    
    train_md, test_md = list(split_data_from_dataframe(flows_dataset[i], 0.8, 42))
    
    flows_train.append(train_md)
    flows_test.append(test_md)

for i in range(len(isomer_labels)):
    print('Isomer {:d} has {:d} samples.'.format(isomer_labels[i], flows_test[i].shape[0]))

# load models

path_flow_models = get_project_path() + '/' + args.flow_type

flows_dic = [load_pickle_file("results_adaptive_is{:d}_{:d}/adaptive_sampling_is{:d}_{:d}.pkl".format(
                            isomer_labels[i],
                            args.flows_id[i],
                            isomer_labels[i],
                            args.flows_id[i]), 
                            path=path_flow_models)['dict_flows'][args.slice][0] 
                            for i in range(len(isomer_labels))]

flow_models = [flows_dic[i]['model'] for i in range(len(isomer_labels))]


if 'mlp' in args.energy_type:
    print('if MLP')
    if args.flow_type == '1-adaptive' and args.mlps_id is not None:
        
        path_mlp_models = get_project_path() + '/0-train-mlp'                                                                      
        mlps_dic = [load_pickle_file("results_mlp_is{:d}_{:d}/is{:d}_mlp_dic_training_{:d}.pkl".format(
                                isomer_labels[i],
                                args.mlps_id[i],
                                isomer_labels[i],
                                args.mlps_id[i]), 
                                path=path_mlp_models) 
                                for i in range(len(isomer_labels))]

        mlp_models = [mlps_dic[i]['model'] for i in range(len(isomer_labels))]
    
    elif args.flow_type == '2-adaptive-mlp':

        mlps_dic = [load_pickle_file("results_adaptive_is{:d}_{:d}/adaptive_sampling_is{:d}_{:d}.pkl".format(
                            isomer_labels[i],
                            args.flows_id[i],
                            isomer_labels[i],
                            args.flows_id[i]), 
                            path=path_flow_models)['dict_mlps'][-1][0]
                            for i in range(len(isomer_labels))]

        mlp_models = [mlps_dic[i]['model'] for i in range(len(isomer_labels))]

        if args.train_mlp_models:

            xs_mlp_train = []
            xs_mlp_test = []

            for i in range(len(isomer_labels)):
            
                datasets = load_pickle_file("results_adaptive_is{:d}_{:d}/adaptive_sampling_is{:d}_{:d}.pkl".format(
                            isomer_labels[i],
                            args.flows_id[i],
                            isomer_labels[i],
                            args.flows_id[i]), 
                            path=path_flow_models)['mlps_datasets'][0]
                #print(len(datasets['mlps_datasets'][0]),
                #      datasets['mlps_datasets'][0]['train'].shape,
                #      datasets['mlps_datasets'][0]['test'].shape)
                
                xs_mlp_train.append(datasets['train'][0])
                xs_mlp_test.append(datasets['test'][0])

                print('Shape of xs_mlp_train: ', xs_mlp_train[i].shape)
                print('Shape of xs_mlp_test: ', xs_mlp_test[i].shape)

                mlp_hyperparams = {'n_iter': args.mlp_n_iter,
                                    'lr': args.mlp_learning_rate,
                                    'bs': args.mlp_batch_size,
                                    'use_scheduler': args.mlp_use_scheduler,
                                    'step_scheduler': args.mlp_step_scheduler,
                                    'save_splits': 1,
                                    }

        else:
    
            xs_mlp_train = None
            xs_mlp_test = None
            mlp_hyperparams = None
    
    else:
        
        raise ValueError('Flow type not recognized for NPs.')

else:

    print('No MLPs')
    mlp_models = [None]*len(isomer_labels)

# initizalize mcmc chains

xs_init = torch.cat(flows_test).clone()
xs_init = xs_init[torch.randperm(xs_init.shape[0])]
xs_init = xs_init[:n_chains]

print('xs_init shape: ', xs_init.shape)

if len(isomer_labels) == 1:

    mixture = False
    simulation_name = 'is{:d}'.format(isomer_labels[0])
    mixture_model = flows_dic[0]['model']

elif len(isomer_labels) > 1:

    mixture = True
    simulation_name = 'mixture'
    mixture_model = Mixture(flow_models, 
                            torch.tensor([0.5, 0.5]).detach())

folder_to_save_results = 'results_multimodal_sampling_{:s}_{:d}'.format(simulation_name, args.process_id)
path_to_save_results = os.getcwd() + '/' + folder_to_save_results

if rank == 0:
    if not os.path.exists(path_to_save_results):
        os.makedirs(path_to_save_results)
        print('Folder created: ', path_to_save_results)

mpi.world.barrier()

mh = run_metropolis(
    model=mixture_model,
    init=xs_init,
    mlp_models=mlp_models,
    n_steps=n_steps,
    n_chains=n_chains,
    id_run='NA',
    mixture=mixture,
    temperature=args.temperature,
    energy_type=args.energy_type,
    alpha=args.alpha,
    update_weights=args.update_weights,
    scheduler_weights=args.scheduler_weights,
    frac_computed=args.frac_computed,
    folder_name=path_to_save_results+'/DFTComputations_{:d}'.format(args.process_id),
    checkpoints=args.checkpoints,
    train_mlp_models=args.train_mlp_models,
    mlp_init_train=xs_mlp_train,
    mlp_init_test=xs_mlp_test,
    mlp_hyperparams=[mlp_hyperparams]*len(isomer_labels),
    train_mlp_scheduler=args.train_mlp_scheduler,
)

mpi.world.barrier()

date_end = np.array([0])

if rank == 0:
    date_end = np.array([set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))])

comm.broadcast(date_end, 0)
date_end = date_end[0]

mpi.world.barrier()
args.date_end = date_end

args.algorithm = 'multimodal_sampling.py'

argparse_dic = vars(args)
mh['args'] = argparse_dic

# save output to pickle file
save_pickle_file(mh, "{:s}_mcmc_dic_{:d}.pkl".format(simulation_name, args.process_id), path = path_to_save_results)

# save arguments to json file
save_json_args(args, 'multimodal_sampling', args.process_id, path_to_save_results)

# plot acceptance ratio

accs = mh['accs'].float().mean(dim=1).detach()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
set_plot_sequential_data(accs, ax=ax, avg=True, window_size=25, color='blue', label='Acceptance ratio')
ax.set_xlabel('Iteration')
ax.set_ylabel('Acceptance ratio')
plt.savefig(path_to_save_results + '/' + 'acceptance_ratio_{:d}.png'.format(args.process_id), dpi=300)

xss = mh['xs']
isomerss = mh['isomers']
cvss = get_collective_variables_from_xs(xss, isomerss)

save_pickle_file(cvss, 
                'cvs_{:s}_{:d}.pkl'.format(simulation_name, args.process_id), 
                path=path_to_save_results)