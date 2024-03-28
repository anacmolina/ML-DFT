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
    get_project_path,
    load_csv_file,
    load_pickle_file,
    save_pickle_file,
    save_json_args,
    set_str_date_to_int,
)
# units
from ase.units import kB
# data handling
from flonacomldft.utils.data_processing import split_data_from_dataframe
# nf training
from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow
# sampling and training
from flonacomldft.full_adaptive_sampling import run_adaptive_sampling
# plotting
from flonacomldft.utils.plots import create_report
# diagnostics
from flonacomldft.utils.diagnostics import get_participation_ratio_from_nlls
# collective variables
from flonacomldft.internal_coordinates import get_collective_variables_from_xs

# parallelization set up
import gpaw.mpi as mpi

print('Adaptive sampling')

ranks = np.arange(0, mpi.world.size)
rank = mpi.rank
comm = mpi.world.new_communicator(ranks)

print('Ranks: ', mpi.world.size)

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

print('Random seed: ', num_seed)
print('Date_start: ', date_start)

# define arpase arguments
parser = argparse.ArgumentParser(description='Prepare experiment')
# execution params
parser.add_argument('-threads', '--threads', type=int, default=None)
parser.add_argument('-pid', '--process-id', type=int, default=date_start)
parser.add_argument('-rs', '--random-seed', type=int, default=num_seed)
parser.add_argument('-path', '--folder-path', type=str, default='andersen')
parser.add_argument('-dataset', '--dataset', type=str, default='mlp')
parser.add_argument('-Nmd', '--N-md-points', type=int, default=500)
parser.add_argument('-Nrd', '--N-random-points', type=int, default=500)
# training params
parser.add_argument('-isomer', '--isomer-label', type=int, nargs='+', default=[0])
# flow params
parser.add_argument('-fni', '--flow-n-iter', type=int, default=100)
parser.add_argument('-flr', '--flow-learning-rate', type=float, default=1e-4)
parser.add_argument('-fbs', '--flow-batch-size', type=int, default=500)
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-nodes', '--hidden-dim', type=int, default=64)
parser.add_argument('-layers', '--hidden-depth', type=int, default=3)
parser.add_argument('-fus', '--flow-use-scheduler', type=bool, default=False)
parser.add_argument('-fss', '--flow-step-scheduler', type=int, default=100)
# mlp params
parser.add_argument('-ni', '--mlp-n-iter', type=int, default=2500)
parser.add_argument('-lr', '--mlp-learning-rate', type=float, default=1e-5)
parser.add_argument('-bs', '--mlp-batch-size', type=int, default=500)
parser.add_argument('-us', '--mlp-use-scheduler', type=bool, default=False)
parser.add_argument('-ss', '--mlp-step-scheduler', type=int, default=100)
# adaptive sampling params
parser.add_argument('-T', '--temperature', type=float, default=350)
parser.add_argument('-nruns', '--n-runs', type=int, default=5)
parser.add_argument('-nchains', '--n-chains', type=int, default=5)
parser.add_argument('-nsteps', '--n-steps', type=int, default=10)
parser.add_argument('-etype', '--energy-type', type=str, default='dft')
parser.add_argument('-frac', '--frac-computed', type=float, default=0.2)
parser.add_argument('-scfrac', '--scheduler-frac-computed', type=int, default=5)
parser.add_argument('-upfrac', '--update-frac-computed', type=bool, default=True)
parser.add_argument('-tmlp', '--train-mlps', type=bool, default=False)
parser.add_argument('-load', '--load-models', type=bool, default=False)
parser.add_argument('-ftype', '--flow-type', type=str, default=None)
parser.add_argument('-upw', '--update-weights', type=bool, default=True)
parser.add_argument('-schw', '--scheduler-weights', type=int, default=10)
parser.add_argument('-alpha', '--alpha', type=float, default=0.5)
parser.add_argument('-npsID', '--mlps-id', type=int, nargs='+', default=None)
parser.add_argument('-nfsID', '--flows-id', type=int, nargs='+', default=None)
parser.add_argument('-ncycles', '--cycles', type=int, default=None, help='Set window size')
parser.add_argument('-scnps', '--scheduler-train-mlps-models', type=bool, default=True)

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


print('Isomer labels: ', isomer_labels)

# mcmc chains parameters
n_runs = args.n_runs
n_chains = args.n_chains
n_steps = args.n_steps
energy_type = args.energy_type

print('Number of runs: ', n_runs)

# path to the datasets
path_datasets = get_path() + '/' + args.folder_path + '/' + 'datasets'

# real center coordinates

flows_dataset = [load_csv_file('is{:d}_{:s}_train.csv'.format(isomer_labels[i], 
                                                            'flow'), path=path_datasets)[:args.N_md_points, :dim+2]
                                                            for i in range(len(isomer_labels))]
flows_train = []
flows_test = []

for i in range(len(isomer_labels)):
    
    train_md, test_md = list(split_data_from_dataframe(flows_dataset[i], 0.8, 42))
    
    flows_train.append(train_md)
    flows_test.append(test_md)

print('Flow dataset shape: ', flows_train[0].shape)

if args.cycles is None:
    args.cycles = 5

n_train_samples_flow = [ args.n_steps * args.n_chains * args.cycles + flows_train[i].shape[0] for i in range(len(isomer_labels)) ]


if 'mlp' in energy_type:

    # load mlp datasets

    mlps_train = []
    mlps_test = []

    for i in range(len(isomer_labels)):

        #xs_train_md, xs_test_md = list(split_data_from_dataframe(flows_dataset[i], 0.8, 42))

        xs_mlp = load_csv_file('is{:d}_{:s}_train.csv'.format(isomer_labels[i], 
                                                                    args.dataset), path=path_datasets)[:args.N_random_points, :dim+2]
        #xs_test_mlp = load_csv_file('is{:d}_{:s}_test.csv'.format(isomer_labels[i], 
        #                                                            args.dataset), path=path_datasets)[:int(args.N_random_points*0.2), :dim+2]

        xs_train_mlp, xs_test_mlp = list(split_data_from_dataframe(xs_mlp, 0.8, 42))

        print("Random points: ", args.N_random_points, xs_train_mlp.shape, xs_test_mlp.shape)
        xs_train = torch.cat((flows_train[i].clone(), xs_train_mlp) )
        xs_test = torch.cat((flows_test[i].clone(), xs_test_mlp) )

        mlps_train.append(xs_train.clone())
        mlps_test.append(xs_test.clone())

        print('MLP dataset shape: ', xs_train.shape, xs_test.shape)

# whether to use a mixture of flows
if len(isomer_labels)==1:

    mixture = False
    simulation_name = "is{:d}".format(isomer_labels[0])

else:

    mixture = True
    simulation_name = "mixture"

# load mlp model dictionaries
if "mlp" in energy_type:

    path_mlp_models = get_project_path() + '/0-train-mlp'                                                                      
    
    mlps_dic = [load_pickle_file("results_mlp_is{:d}_{:d}/is{:d}_mlp_dic_training_{:d}.pkl".format(
                                isomer_labels[i],
                                args.mlps_id[i],
                                isomer_labels[i],
                                args.mlps_id[i]), 
                                path=path_mlp_models) 
                                for i in range(len(isomer_labels))]
    
else:

    mlps_dic = [None for i in range(len(isomer_labels))]
    mlps_train = [None for i in range(len(isomer_labels))]
    mlps_test = [None for i in range(len(isomer_labels))]

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
        flow_dataset,
        n_iter=args.flow_n_iter,
        lr=args.flow_learning_rate,
        bs=args.flow_batch_size,
        use_scheduler=args.flow_use_scheduler,
        step_scheduler=args.flow_step_scheduler,
        save_splits=1,
        grad_clip=1e4,
        with_tqdm=False,
    ) for model, flow_dataset in zip(models, flows_train)]

else:

    if args.flow_type == 'adaptive':

        path_flow_models = get_project_path() + '/1-adaptive' 
                                        
    elif args.flow_type == 'adaptive-mlp':

        path_flow_models = get_project_path() + '/2-adaptive-mlp'

    else:

        raise ValueError('Flow type not recognized')

    flows_dic = [load_pickle_file("results_adaptive_is{:d}_{:d}/is{:d}_flow_dic_{:d}.pkl".format(
                                isomer_labels[i], 
                                args.flows_id[i], 
                                isomer_labels[i], 
                                args.flows_id[i]), 
                                path=path_flow_models) 
                                for i in range(len(isomer_labels))]

# retraining hyperparameters
flow_hyperparams = {'n_iter': args.flow_n_iter,
    'lr': args.flow_learning_rate,
    'bs': args.flow_batch_size,
    'use_scheduler': args.flow_use_scheduler,
    'step_scheduler': args.flow_step_scheduler,
    'save_splits': 1,
    }


mlp_hyperparams = {'n_iter': args.mlp_n_iter,
    'lr': args.mlp_learning_rate,
    'bs': args.mlp_batch_size,
    'use_scheduler': args.mlp_use_scheduler,
    'step_scheduler': args.mlp_step_scheduler,
    'save_splits': 1,
    }

# path to save results
folder_to_save_results = 'results_adaptive_{:s}_{:d}'.format(simulation_name, args.process_id)
path_to_save_results = os.getcwd() + '/' + folder_to_save_results

if rank == 0:
    
    if not os.path.exists(path_to_save_results):
        
        os.makedirs(path_to_save_results)
        print('Folder created: ', path_to_save_results)

mpi.world.barrier()


# init chains
shuffle = torch.randperm(torch.cat(flows_test).shape[0]) 
mcmc_init = torch.cat(flows_test)[shuffle].clone()[:n_chains] 

if rank == 0:
    time_init = time.time()

if len(isomer_labels) > 1:
    mixture = True

mpi.world.barrier()

print("MLP sizes datasets before adaptive: ", mlps_train[0].shape, mlps_test[0].shape)

# run adaptive sampling
adaptive =run_adaptive_sampling(
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
    scheduler_train_mlp_models=args.scheduler_train_mlps_models,
    frac_computed=args.frac_computed,
    scheduler_frac_computed=args.scheduler_frac_computed,
    update_frac_computed=args.update_frac_computed,
    init_weights=None,
    update_weights=args.update_weights,
    scheduler_weights=args.scheduler_weights,
    alpha=args.alpha,
    n_samples_train_flow=n_train_samples_flow,
    folder_name=path_to_save_results,
    )

if rank == 0:
    date_end =  np.array([set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))])

    args.date_end = date_end[0]
    args.algorithm = 'adaptive_sampling.py'
    args.time_init = time_init

    argparse_dict = vars(args)
    adaptive['args'] = argparse_dict

    save_json_args(args, 'adaptive_sampling', args.process_id, path_to_save_results)

    # save all simulation results in a pickle file
    f = "adaptive_sampling_{:s}_{:d}.pkl".format(simulation_name, args.process_id)
    save_pickle_file(adaptive, f, path = path_to_save_results)

    # save chains conformations, energies and isomers in a csv file
    xs = torch.cat(adaptive["xs"])
    us = torch.cat(adaptive["us"]).reshape((n_runs*n_steps, n_chains, 1))
    isomers = torch.cat(adaptive["isomers"]).reshape((n_runs*n_steps, n_chains, 1))

    data_mcmc = torch.cat((xs, us, isomers), dim=2).detach().numpy()
    data_mcmc = data_mcmc.reshape((n_runs*n_steps*n_chains, dim+2))

    df = pd.DataFrame(data_mcmc, columns=['rc{:d}'.format(i) for i in range(dim)]
                       + ['potential_energy', 'isomer'])
    df.to_csv(path_to_save_results + '/' + 'xs_adaptive_{:s}_{:d}.csv'.format(simulation_name, args.process_id), 
              index=False)

    # save last flow dic
    for i in range(len(isomer_labels)):
        save_pickle_file(adaptive["dict_flows"][-1][i], 
                    "is{:d}_flow_dic_{:d}.pkl".format(isomer_labels[i], args.process_id), 
                    path = path_to_save_results)

    # save acceptance rates and times
    time_mcmc = adaptive["time_mcmc"]
    time_mcmc_flatten = torch.tensor([t - time_init for time_set in time_mcmc 
                                  for t in time_set]).reshape(-1, 1)

    accs_rate = torch.cat(adaptive["accs"]).float().squeeze().mean(dim=1).reshape(-1, 1)
    
    data_accs = torch.cat((accs_rate, time_mcmc_flatten), dim=1)
    df_acc = pd.DataFrame(data_accs, columns=['accs', 'time'])
    
    df_acc.to_csv(path_to_save_results + '/' + 'accs_{:s}_{:d}.csv'.format(simulation_name, args.process_id), 
                  index=False)

    energies = {'md': flows_train[0][:, 12].detach().numpy()}

    xss = torch.cat(adaptive['xs'])
    isomerss = torch.cat(adaptive['isomers'])

    cvss = get_collective_variables_from_xs(xss, isomerss)

    save_pickle_file(cvss, 
                     'cvs_{:s}_{:d}.pkl'.format(simulation_name, args.process_id), 
                     path=path_to_save_results)

    us_proposals = adaptive['us_proposals']
    nlls_proposals = adaptive['nlls_proposals']

    part_ratios = []

    for u, nll in zip(us_proposals, nlls_proposals):

        part_ratios.append(get_participation_ratio_from_nlls(u.flatten(),
                                            nll.flatten(),
                                            kB,
                                            T=args.temperature))

    part_ratios = torch.stack(part_ratios).detach().numpy()

    df_part = pd.DataFrame(part_ratios, columns=['part_ratio'])
    df_part.to_csv(path_to_save_results + '/' + 'part_ratios_{:s}_{:d}.csv'.format(simulation_name, args.process_id), 
                  index=False)

    accs = torch.cat(adaptive['accs']).float().mean(dim=1).detach().numpy()
    us = torch.cat(adaptive['us']).flatten().detach().numpy()
    losses = torch.cat([torch.tensor(frame[0]['train_losses']) for frame in adaptive['dict_flows']]).detach().numpy()
    nlls = torch.cat(adaptive['nlls']).flatten().detach().numpy()
    
    energies['adaptive'] = us
    
    plot_data = {'accs': accs,
                 'energies': energies,
                 'losses': losses,
                 'nlls': nlls,
                 'cvs': cvss.reshape(-1, 2),
                 'part_ratios': part_ratios}

    
    fig, axs = create_report(plot_data)
    
    filename = path_to_save_results + '/' + 'report_adaptive_{:s}_{:d}.png'.format(simulation_name, args.process_id)
    
    fig.suptitle('Adaptive sampling', fontsize=16)
    
    fig.savefig(filename)
