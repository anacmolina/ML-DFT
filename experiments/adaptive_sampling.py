import warnings
warnings.filterwarnings("ignore")

from ase.parallel import parprint as print

# standard library imports
import os
import time
import argparse

# scientific library imports
import torch
import numpy as np
import pandas as pd

# flonaco imports
# io handling
from flonacomldft.utils.io_utils import (
    get_path,
    load_csv_file,
    load_pickle_file,
    save_pickle_file,
    save_json_args,
    set_str_date_to_int,
)
# coordinates handling
from flonacomldft.internal_coordinates import Coordinates_mapping
# nf training
from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow
# sampling and training
from flonacomldft.full_adaptive_sampling import run_adaptive_sampling
# plotting
from flonacomldft.utils.plots import Adaptive_Plotter, create_report

from flonacomldft.utils.data_processing import split_data_from_dataframe


# parallelization set up
import gpaw.mpi as mpi

ranks = np.arange(0, mpi.world.size)
rank = mpi.rank
comm = mpi.world.new_communicator(ranks)

print('ranks: ', mpi.world.size)

num_seed = np.array([0])
date_start = np.array([0])

# only rank 0 generates the seed and date_start
if rank == 0:
    num_seed = np.random.randint(0, 100, (1,))
    date_start = np.array([set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))])

mpi.world.barrier()

comm.broadcast(num_seed, 0)
comm.broadcast(date_start, 0)

num_seed = num_seed[0]
date_start = date_start[0]

print('seed: ', num_seed, rank)
print('date_start: ', date_start, rank)

# define arpase arguments
parser = argparse.ArgumentParser(description='Prepare experiment')
# execution params
parser.add_argument('-threads', '--threads', type=int, default=None)
parser.add_argument('-pid', '--process-id', type=int, default=date_start)
parser.add_argument('-rs', '--random-seed', type=int, default=num_seed)
parser.add_argument('-path', '--folder-path', type=str, default='andersen')
parser.add_argument('-dataset', '--dataset', type=str, default='mlp')
parser.add_argument('-N', '--N', type=int, default=5000)
# training params
parser.add_argument('-isomer', '--isomer-label', type=int, nargs='+', default=[0])
# flow params
parser.add_argument('-fni', '--flow-n-iter', type=int, default=10)
parser.add_argument('-flr', '--flow-learning-rate', type=float, default=1e-4)
parser.add_argument('-fbs', '--flow-batch-size', type=int, default=100)
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-nodes', '--hidden-dim', type=int, default=64)
parser.add_argument('-layers', '--hidden-depth', type=int, default=3)
parser.add_argument('-fus', '--flow-use-scheduler', type=bool, default=False)
parser.add_argument('-fss', '--flow-step-scheduler', type=int, default=100)
# mlp params
parser.add_argument('-ni', '--mlp-n-iter', type=int, default=10)
parser.add_argument('-lr', '--mlp-learning-rate', type=float, default=1e-4)
parser.add_argument('-bs', '--mlp-batch-size', type=int, default=500)
parser.add_argument('-us', '--mlp-use-scheduler', type=bool, default=False)
parser.add_argument('-ss', '--mlp-step-scheduler', type=int, default=100)
# adaptive sampling params
parser.add_argument('-T', '--temperature', type=float, default=350)
parser.add_argument('-nruns', '--n-runs', type=int, default=5)
parser.add_argument('-nchains', '--n-chains', type=int, default=5)
parser.add_argument('-nsteps', '--n-steps', type=int, default=10)
parser.add_argument('-etype', '--energy-type', type=str, default='emt')
parser.add_argument('-frac', '--frac-computed', type=float, default=0.5)
parser.add_argument('-tmlp', '--train-mlps', type=bool, default=False)
parser.add_argument('-load', '--load-models', type=bool, default=False)
parser.add_argument('-upw', '--update-weights', type=bool, default=True)
parser.add_argument('-schw', '--scheduler-weights', type=int, default=10)
parser.add_argument('-alpha', '--alpha', type=float, default=0.5)

args = parser.parse_args()
args.date_start = str(date_start)

# torch settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.threads is not None:
    torch.set_num_threads(args.threads)

print('Device: ', device)
print('Threads: ', torch.get_num_threads(), rank)

# set random seed
torch.manual_seed(int(args.random_seed))
mpi.world.barrier()

# isomer labels
isomer_labels = args.isomer_label
dim = 12

# mcmc chains parameters
n_runs = args.n_runs
n_chains = args.n_chains
n_steps = args.n_steps
energy_type = args.energy_type

# path to the datasets
path_datasets = get_path() + '/' + args.folder_path + '/' + 'datasets'

# real center coordinates

flows_dataset = [load_csv_file('is{:d}_{:s}_train.csv'.format(isomer_labels[i], 
                                                            'flow'), path=path_datasets)[:args.N, :dim+2]
                                                            for i in range(len(isomer_labels))]

flows_train = []
flows_test = []

mlps_train = []
mlps_test = []

for i in range(len(isomer_labels)):

    xs_train_md, xs_test_md = list(split_data_from_dataframe(flows_dataset[i], 0.8, 42))

    flows_train.append(xs_train_md.clone())
    flows_test.append(xs_test_md.clone())

    xs_train_mlp = load_csv_file('is{:d}_{:s}_train.csv'.format(isomer_labels[i], 
                                                                args.dataset), path=path_datasets)[:, :dim+2]
    xs_test_mlp = load_csv_file('is{:d}_{:s}_test.csv'.format(isomer_labels[i], 
                                                                args.dataset), path=path_datasets)[:, :dim+2]

    xs_train = torch.cat((xs_train_md, xs_train_mlp) )
    xs_test = torch.cat((xs_test_md, xs_test_mlp) )

    mlps_train.append(xs_train.clone())
    mlps_test.append(xs_test.clone())


for i in range(len(isomer_labels)):
    print('flow_train shape: ', flows_train[i].shape, isomer_labels[i], rank)

# whether to use a mixture of flows
if len(isomer_labels)==1:

    mixture = False
    simulation_name = "is{:d}".format(isomer_labels[0])

else:

    mixture = True
    simulation_name = "mixture"

# load mlp model dictionaries
if "mlp" in energy_type: 
    path_models = get_path() + '/' + args.folder_path + '/' + 'models'

    mlps_dic = [load_pickle_file("dict_mlp_model_is{:d}_{:s}.pkl".format(
                                isomer_labels[i],
                                args.dataset), 
                                path=path_models) 
                                for i in range(len(isomer_labels))]
    
else:

    mlps_dic = [None for i in range(len(isomer_labels))]

# path to save results
folder_to_save_results = 'results_adaptive_{:s}_{:d}'.format(simulation_name, args.process_id)
path_to_save_results = os.getcwd() + '/' + folder_to_save_results

if rank == 0:
    
    if not os.path.exists(path_to_save_results):
        
        os.makedirs(path_to_save_results)
        print('folder created: ', path_to_save_results, rank)

mpi.world.barrier()

if args.load_models==False:

    #set covariance matrix for flows

    cov = [torch.cov(flows_train[i][:, :dim].T).detach() + 1e-5 * torch.eye(flows_train[i][:, :dim].shape[1]).detach() 
           for i in range(len(isomer_labels))]

    models = [RealNVP_MLP(dim=flows_train[i][:, :dim].shape[1],
                        n_blocks=args.n_blocks,
                        block_depth=1,
                        init_weight_scale=1e-3,
                        base_cov=cov[i],
                        hidden_dim=args.hidden_dim,
                        hidden_depth=args.hidden_depth,
                        device=device,
                        )
                        for i in range(len(isomer_labels))]

    # training flow model
    flows_dic = [train_flow(
        model,
        flow_train,
        n_iter=args.flow_n_iter,
        lr=args.flow_learning_rate,
        bs=args.flow_batch_size,
        use_scheduler=args.flow_use_scheduler,
        step_scheduler=args.flow_step_scheduler,
        save_splits=1,
        grad_clip=1e4,
        with_tqdm=False,
    ) for model, flow_train in zip(models, flows_train)]

else:

    path_models = get_path() + '/' + args.folder_path + '/' + 'models'

    flows_dic = [load_pickle_file("dict_flow_model_is{:d}_{:s}.pkl".format(
                                isomer_labels[i], energy_type), 
                                path=path_models) 
                                for i in range(len(isomer_labels))]

# retraining hyperparameters
flow_hyperparams = {'n_iter': args.flow_n_iter,
    'lr': args.flow_learning_rate,
    'bs': args.flow_batch_size,
    'use_scheduler': args.flow_use_scheduler,
    'step_scheduler': args.flow_step_scheduler,
    'save_splits': 1,
    }

print("flows size: ", len(flows_dic))

mlp_hyperparams = {'n_iter': args.mlp_n_iter,
    'lr': args.mlp_learning_rate,
    'bs': args.mlp_batch_size,
    'use_scheduler': args.mlp_use_scheduler,
    'step_scheduler': args.mlp_step_scheduler,
    'save_splits': 1,
    }

print("mlps size: ", len(mlps_dic))

# init chains
shuffle = torch.randperm(torch.cat(flows_test).shape[0]) # this works only for one isomer
mcmc_init = torch.cat(flows_test)[shuffle].clone()[:n_chains] # TODO: generalize for more isomers

print('init_mcmc.shape: ', mcmc_init.shape, rank)
print('init_mcmc: ', mcmc_init, rank)

if rank == 0:
    time_init = time.time()

if len(isomer_labels) > 1:
    mixture = True

mpi.world.barrier()
# run adaptive sampling
# TODO: include results folder and filename argument for mcmc
out =run_adaptive_sampling(
    mcmc_init = mcmc_init,
    n_chains=n_chains,
    n_steps=n_steps,
    n_runs=n_runs,
    flow_init_train=flows_train,
    dict_flows_init=flows_dic,
    flow_hyperparams=[flow_hyperparams, flow_hyperparams],
    energy_type=energy_type,
    temperature=args.temperature,
    mixture=mixture,
    dim=dim,
    dict_mlps_init=mlps_dic,
    mlp_init_train=mlps_train,
    mlp_init_test=mlps_test,
    mlp_hyperparams=[mlp_hyperparams, mlp_hyperparams],
    train_mlp_models=args.train_mlps,
    frac_computed=args.frac_computed,
    init_weights=None,
    update_weights=args.update_weights,
    scheduler_weights=args.scheduler_weights,
    alpha=args.alpha,
    n_samples_train_flow=None,
    folder_name=path_to_save_results,
    )

date_end = np.array([0])

if rank == 0:
    date_end =  np.array([set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))])

    args.date_end = date_end[0]
    args.algorithm = 'adaptive_sampling.py'
    args.time_init = time_init

    argparse_dict = vars(args)
    out['args'] = argparse_dict

    save_json_args(args, 'adaptive_sampling', args.process_id, path_to_save_results)

    # save all simulation results in a pickle file
    f = "adaptive_sampling_{:s}_{:d}.pkl".format(simulation_name, args.process_id)
    save_pickle_file(out, f, path = path_to_save_results)

    # save chains conformations, energies and isomers in a csv file
    # xs = torch.stack(out["xs"])
    # us = torch.stack(out["us"]).squeeze()
    # accs = torch.stack(out["accs"]).squeeze()
    # isomers = torch.stack(out["isomers"]).squeeze()
# 
    # xs_chains = xs.reshape(xs.shape[0]*xs.shape[1], xs.shape[2], xs.shape[3])
    # us_chains = us.reshape(us.shape[0]*us.shape[1], us.shape[2])
    # accs_chains = accs.reshape(accs.shape[0]*accs.shape[1], accs.shape[2])
    # isomers_chains = isomers.reshape(isomers.shape[0]*isomers.shape[1], isomers.shape[2])

    #data_mcmc = torch.cat([torch.cat((xs_chains[:, i, :12], us_chains[:, i].reshape(-1, 1), isomers_chains[:, i].reshape(-1, 1)), dim=1) for i in range(out["args"]["n_chains"])])
#
    #df = pd.DataFrame(data_mcmc.detach().numpy(), columns=['b-2-0', 'b-3-2', 'b-4-2', 'b-5-4', 'b-1-4', 'a-3-2-0', 'a-4-2-3',
    #   'a-5-4-2', 'a-1-4-2', 'd-4-2-3-0', 'd-5-4-2-3', 'd-1-4-2-3',
    #   'potential_energy', 'isomer',])
#
    #df.to_csv(path_to_save_results + '/' + 'MCMC_{:s}_{:d}.csv'.format(simulation_name, args.process_id), index=False)

    # save last flow model
    # save_pickle_file(out["dict_flows_training"][-1][0]["model"], 
    #                  "flow_model_{:s}_{:d}.pkl".format(simulation_name, args.process_id), path = path_to_save_results)

    # save acceptance rates and times

    # time_mcmc = [mcmc['time_mcmc'] for mcmc in mcmc_runs]
    # time_mcmc_flatten = [t - time_init for time_set in time_mcmc for t in time_set]

    #accs = torch.stack(out["accs"]).squeeze()
    #time_mcmcs = torch.stack(out["time_mcmc"]).squeeze()
    #print(accs.shape, time_mcmcs)
    #accs_flatten = accs.reshape(accs.shape[0]*accs.shape[1], accs.shape[2])
#
    #data_accs = torch.cat((torch.sta(out['time_mcmc']).reshape(-1, 1), accs_flatten.mean(dim=1).reshape(-1, 1), accs_flatten), dim=1)
    #df_acc = pd.DataFrame(data_accs, columns=['time', 'accs'] + ['chain_{:d}'.format(i) for i in range(accs_flatten.shape[1])])
#
    #df_acc.to_csv(path_to_save_results + '/' + 'accs_{:s}_{:d}.csv'.format(simulation_name, args.process_id), index=False)
#
    #adaptive_plotter = Adaptive_Plotter(out)
#
    #energies = {'train': flows_train[0][:, 12].detach().numpy(),}
#
    #create_report(adaptive_plotter, energies=energies, path=path_to_save_results + '/')
