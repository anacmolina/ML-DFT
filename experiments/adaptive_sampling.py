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
from flonacomldft.full_adaptive_sampling import adaptive_sampling
# plotting
from flonacomldft.utils.plots import Adaptive_Plotter, create_report

from flonacomldft.utils.data_processing import load_datasets


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
parser.add_argument('-path', '--folder-path', type=str, default='emt_berendsen')
# training params
parser.add_argument('-isomer', '--isomer-label', type=int, nargs='+', default=[0])
#parser.add_argument('-ids', '--ids', type=int, nargs='+', default=[None, None])
# flow params
parser.add_argument('-fni', '--flow-n-iter', type=int, default=5)
parser.add_argument('-flr', '--flow-learning-rate', type=float, default=1e-4)
parser.add_argument('-fbs', '--flow-batch-size', type=int, default=100)
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-nodes', '--hidden-dim', type=int, default=64)
parser.add_argument('-layers', '--hidden-depth', type=int, default=3)
parser.add_argument('-fus', '--flow-use-scheduler', type=bool, default=False)
parser.add_argument('-fss', '--flow-step-scheduler', type=int, default=100)
parser.add_argument('-ratios', '--do-ratios', type=bool, default=False)
parser.add_argument('-prop', '--n-prop', type=int, default=50)
# mlp params
parser.add_argument('-ni', '--mlp-n-iter', type=int, default=10)
parser.add_argument('-lr', '--mlp-learning-rate', type=float, default=1e-5)
parser.add_argument('-bs', '--mlp-batch-size', type=int, default=500)
parser.add_argument('-us', '--mlp-use-scheduler', type=bool, default=False)
parser.add_argument('-ss', '--mlp-step-scheduler', type=int, default=100)
# adaptive sampling params
parser.add_argument('-T', '--temperature', type=float, default=350)
parser.add_argument('-nruns', '--n-runs', type=int, default=5)
parser.add_argument('-nchains', '--n-chains', type=int, default=5)
parser.add_argument('-nsteps', '--n-steps', type=int, default=10)
parser.add_argument('-etype', '--energy-type', type=str, default='emt')
parser.add_argument('-frac', '--frac-dft', type=float, default=0.5)
parser.add_argument('-rmlp', '--retrain-mlps', type=bool, default=False)


args = parser.parse_args()
args.date_start = str(date_start)

# torch settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.threads is not None:
    torch.set_num_threads(args.threads)

print('device: ', device)
print('threads: ', torch.get_num_threads(), rank)

# set random seed
torch.manual_seed(int(args.random_seed))
mpi.world.barrier()

# isomer labels
isomer_labels = args.isomer_label

# mcmc chains parameters
n_runs = args.n_runs
n_chains = args.n_chains
n_steps = args.n_steps
energy_type = args.energy_type

# path to the datasets
path_datasets = get_path() + '/' + args.folder_path + '/' + 'datasets'

# real center coordinates
flow_xs_train = []
flow_xs_test = []

for isomer_label in isomer_labels:
    
    dataset = load_datasets(args.folder_path, isomer_label, name='flow', real_centered=True)
    flow_xs_train.append(dataset['train'].clone())
    flow_xs_test.append(dataset['test'].clone())

    del dataset


for i in range(len(isomer_labels)):
    print('flow_xs_train.shape: ', flow_xs_train[i].shape, isomer_labels[i], rank)
    print('flow_xs_test.shape: ', flow_xs_test[i].shape, isomer_labels[i], rank)

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

    mlps_dic = [load_pickle_file("dict_mlp_model_is{:d}.pkl".format(
                                isomer_labels[i]), 
                                path=path_models) 
                                for i in range(len(isomer_labels))]
    
    # mlps_dic = [{'model': load_pickle_file("mlp_model_is{:d}.pkl".format(
    #                             isomer_labels[i]), 
    #                             path=path_models)}
    #                             for i in range(len(isomer_labels))]
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

#set covariance matrix for flows

cov = [torch.cov(flow_xs_train[i][:, :12].T).detach() + 1e-5 * torch.eye(flow_xs_train[i][:, :12].shape[1]).detach() for i in range(len(isomer_labels))]
#torch.cov(xs_train.T).detach() + 1e-5 * torch.eye(xs_train.shape[1]).detach()

models = [RealNVP_MLP(dim=flow_xs_train[i][:, :12].shape[1],
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
    train,
    test,
    n_iter=args.flow_n_iter,
    lr=args.flow_learning_rate,
    batch_size=args.flow_batch_size,
    use_scheduler=False,
    step_schedule=args.flow_step_scheduler,
    save_splits=10,
    grad_clip=1e4,
    with_tqdm=False,
    compute_part_ratio=args.do_ratios,
    energy_type=args.energy_type,
    n_prop=args.n_prop,
    path=path_to_save_results,
    mlp_model=mlps_dic[i],
) for model, train, test in zip(models, flow_xs_train, flow_xs_test)]

# retraining hyperparameters
flow_hyperparams = {'n_iter': args.flow_n_iter,
    'lr': args.flow_learning_rate,
    'batch_size': args.flow_batch_size,
    'use_scheduler': args.flow_use_scheduler,
    'step_schedule': args.flow_step_scheduler,

    'energy_type': args.energy_type,
    'compute_part_ratio': args.do_ratios,
    'n_prop': args.n_prop,

    'save_splits': 10,
    'grad_clip': 1e4,
    }

print("flows size: ", len(flows_dic))

mlp_hyperparams = {'n_iter': args.mlp_n_iter,
    'lr': args.mlp_learning_rate,
    'batch_size': args.mlp_batch_size,

    'use_scheduler': args.mlp_use_scheduler,
    'step_schedule': args.mlp_step_scheduler,

    'save_splits': 10,
    }

print("mlps size: ", len(mlps_dic))

# init chains
shuffle = torch.randperm(torch.cat(flow_xs_test).shape[0]) # this works only for one isomer
init_mcmc = torch.cat(flow_xs_test)[shuffle].clone()[:n_chains] # TODO: generalize for more isomers

print('init_mcmc.shape: ', init_mcmc.shape, rank)
print('init_mcmc: ', init_mcmc, rank)

if rank == 0:
    time_init = time.time()

mpi.world.barrier()
# run adaptive sampling
# TODO: include results folder and filename argument for mcmc
out = adaptive_sampling(
    flow_init_train=flow_xs_train,
    flow_init_test=flow_xs_test,
    init_mcmc=init_mcmc,
    n_runs=n_runs,
    n_chains=n_chains,
    n_steps=n_steps,
    energy_type=energy_type,
    dict_flows_init=flows_dic,
    flow_hyperparams=[flow_hyperparams, flow_hyperparams], # TODO: generalize for more flows
    dict_mlps_init=mlps_dic,
    mlp_hyperparams=[mlp_hyperparams, mlp_hyperparams], # TODO: generalize for more flows
    path=path_to_save_results,
    save_ratios = 1,
    retrain_mlps=args.retrain_mlps,
    frac_dft=args.frac_dft,
    T=args.temperature,
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

    # save all normalizing flows in a pickle file
    save_pickle_file(out["dict_flows_training"], "all_flows_{:s}_{:d}.pkl".format(simulation_name, args.process_id), path = path_to_save_results)

    # save all mcmc runs in a pickle file
    save_pickle_file(out["mcmc_runs"], "all_mcmc_runs_{:s}_{:d}.pkl".format(simulation_name, args.process_id), path = path_to_save_results)

    # save chains conformations, energies and isomers in a csv file
    xs = torch.stack(out["xs"])
    us = torch.stack(out["us"]).squeeze()
    accs = torch.stack(out["accs"]).squeeze()
    isomers = torch.stack(out["isomers"]).squeeze()

    xs_chains = xs.reshape(xs.shape[0]*xs.shape[1], xs.shape[2], xs.shape[3])
    us_chains = us.reshape(us.shape[0]*us.shape[1], us.shape[2])
    accs_chains = accs.reshape(accs.shape[0]*accs.shape[1], accs.shape[2])
    isomers_chains = isomers.reshape(isomers.shape[0]*isomers.shape[1], isomers.shape[2])

    data_mcmc = torch.cat([torch.cat((xs_chains[:, i, :12], us_chains[:, i].reshape(-1, 1), isomers_chains[:, i].reshape(-1, 1)), dim=1) for i in range(out["args"]["n_chains"])])

    df = pd.DataFrame(data_mcmc.detach().numpy(), columns=['b-2-0', 'b-3-2', 'b-4-2', 'b-5-4', 'b-1-4', 'a-3-2-0', 'a-4-2-3',
       'a-5-4-2', 'a-1-4-2', 'd-4-2-3-0', 'd-5-4-2-3', 'd-1-4-2-3',
       'potential_energy', 'isomer',])

    df.to_csv(path_to_save_results + '/' + 'MCMC_{:s}_{:d}.csv'.format(simulation_name, args.process_id), index=False)

    # save last flow model
    save_pickle_file(out["dict_flows_training"][-1][0]["model"], 
                     "flow_model_{:s}_{:d}.pkl".format(simulation_name, args.process_id), path = path_to_save_results)

    # save acceptance rates and times
    mcmc_runs = out["mcmc_runs"]

    time_mcmc = [mcmc['time_mcmc'] for mcmc in mcmc_runs]
    time_mcmc_flatten = [t - time_init for time_set in time_mcmc for t in time_set]

    accs = torch.stack(out["accs"]).squeeze()
    accs_flatten = accs.reshape(accs.shape[0]*accs.shape[1], accs.shape[2])

    data_accs = torch.cat((torch.tensor(time_mcmc_flatten).reshape(-1, 1), accs_flatten.mean(dim=1).reshape(-1, 1), accs_flatten), dim=1)
    df_acc = pd.DataFrame(data_accs, columns=['time', 'accs'] + ['chain_{:d}'.format(i) for i in range(accs_flatten.shape[1])])

    df_acc.to_csv(path_to_save_results + '/' + 'accs_{:s}_{:d}.csv'.format(simulation_name, args.process_id), index=False)

    # save participation ratios
    if args.do_ratios:
        part_ratios = torch.stack([out["dict_flows_training"][i][0]["part_ratios"] for i in range(len(out["dict_flows_training"]))])

        time_part_ratios = [out["dict_flows_training"][i][0]["time_part_ratios"] for i in range(len(out["dict_flows_training"]))]
        time_part_ratios_flatten = torch.tensor([t-time_init for time_set in time_part_ratios for t in time_set])

        data_part_ratios = torch.stack((part_ratios.flatten(), time_part_ratios_flatten), dim=1)

        df_part_ratios = pd.DataFrame(data_part_ratios.detach(), columns=['part_ratios', 'time'])
        df_part_ratios.to_csv(path_to_save_results + '/' + 'part_ratios_{:s}_{:d}.csv'.format(simulation_name, args.process_id), index=False)

    adaptive_plotter = Adaptive_Plotter(out)

    energies = {'train': flow_xs_train[0][:, 12].detach().numpy(),}

    create_report(adaptive_plotter, energies=energies, path=path_to_save_results + '/')
