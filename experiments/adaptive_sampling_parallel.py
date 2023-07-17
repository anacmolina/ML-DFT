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

# flonaco imports
# io handling
from flonacomldft.utils.io_utils import (
    get_path,
    load_csv_file,
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

# parallelization set up
import gpaw.mpi as mpi

ranks = np.arange(0, mpi.world.size)
rank = mpi.rank
comm = mpi.world.new_communicator(ranks)

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
parser.add_argument('-path', '--folder-path', type=str, default='emt_berendsen/')
# training params
parser.add_argument('-isomer', '--isomer-label', type=int, nargs='+', default=[0])
#parser.add_argument('-ids', '--ids', type=int, nargs='+', default=[None, None])
# flow params
parser.add_argument('-ni', '--n-iter', type=int, default=10)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-bs', '--batch-size', type=int, default=512)
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-nodes', '--hidden-dim', type=int, default=64)
parser.add_argument('-layers', '--hidden-depth', type=int, default=3)
parser.add_argument('-us', '--use-scheduler', type=bool, default=False)
parser.add_argument('-ratios', '--do-ratios', type=bool, default=False)
parser.add_argument('-prop', '--n-prop', type=int, default=50)
# adaptive sampling params
parser.add_argument('-nruns', '--n-runs', type=int, default=5)
parser.add_argument('-nchains', '--n-chains', type=int, default=5)
parser.add_argument('-nsteps', '--n-steps', type=int, default=10)
parser.add_argument('-etype', '--energy-type', type=str, default='emt')

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
coord_mapping = Coordinates_mapping(etype=energy_type)

# TODO: is this the best way to get the dataset?
# TODO: real center data in a file already, add collective varibles to csv file 
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

flow_xs_train, flow_zmat_train = get_dataset('flow_train', path_datasets, isomer_labels)
flow_xs_test, flow_zmat_test = get_dataset('flow_test', path_datasets, isomer_labels)

for i in range(len(isomer_labels)):
    print('flow_xs_train.shape: ', flow_xs_train[i].shape, isomer_labels[i], rank)
    print('flow_xs_test.shape: ', flow_xs_test[i].shape, isomer_labels[i], rank)

# whether to use a mixture of flows
if len(isomer_labels)==1:
    mixture = False

    if "mlp" in energy_type: 
        ## ADD MLP MODEL
        pass
        #mlp_models = mlp_models[0]
else:
    mixture = True

if mixture:
    simulation_name = "mixture"
else:
    simulation_name = "is{:d}".format(isomer_labels[0])

# path to save results
folder_to_save_results = 'results_adaptive_{:s}_{:d}'.format(simulation_name, args.process_id)
path_to_save_results = os.getcwd() + '/' + folder_to_save_results
if rank == 0:
    if not os.path.exists(path_to_save_results):
        os.makedirs(path_to_save_results)
        print('folder created: ', path_to_save_results, rank)

mpi.world.barrier()

cov = [torch.cov(flow_xs_train[i][:, :12].T).detach() + 1e-5 * torch.eye(flow_xs_train[i][:, :12].shape[1]).detach() for i in range(len(isomer_labels))]
print('cov.shape: ', len(cov))
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
    n_iter=args.n_iter,
    lr=args.learning_rate,
    batch_size=args.batch_size,
    use_scheduler=False,
    step_schedule=args.n_iter/5,
    save_splits=10,
    grad_clip=1e4,
    with_tqdm=False,
    compute_part_ratio=args.do_ratios,
    energy_type=args.energy_type,
    n_prop=args.n_prop,
    path=path_to_save_results,
    #N_samples=8000,
) for model, train, test in zip(models, flow_xs_train, flow_xs_test)]

mlps_dic = None

# retraining hyperparameters
flow_hyperparams = {'n_iter': args.n_iter,
    'lr': args.learning_rate,
    'use_scheduler': False,
    'step_schedule': 100,
    'save_splits': 10,
    'grad_clip': 1e4,
    'compute_part_ratio': args.do_ratios,
    'energy_type': args.energy_type,
    'batch_size': args.batch_size,
    'n_prop': args.n_prop,
    'use_scheduler': args.use_scheduler,
    }

# init chains
shuffle = torch.randperm(flow_xs_test[0].shape[0]) # this works only for one isomer
init_mcmc = flow_xs_test[0][shuffle].clone()[:n_chains] # TODO: generalize for more isomers

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
    path=path_to_save_results,
    save_ratios = 5,
    #seed=args.random_seed,
    )

date_end = np.array([0])

if rank == 0:
    date_end =  np.array([set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))])

#mpi.world.barrier()
#comm.broadcast(date_end, 0)
#if rank == 0:
    args.date_end = date_end[0]
    args.algorithm = 'adaptive_sampling.py'

    argparse_dict = vars(args)
    out['args'] = argparse_dict

    save_json_args(args, 'adaptive_sampling', args.process_id, path_to_save_results)

    f = "adaptive_sampling_{:s}_{:d}.pkl".format(simulation_name, args.process_id)
    save_pickle_file(out, f, path = path_to_save_results)

    from flonacomldft.utils.plots import Adaptive_Plotter, create_report

    adaptive_plotter = Adaptive_Plotter(out)

    energies = {'train': flow_xs_train[0][:, 12].detach().numpy(),}

    create_report(adaptive_plotter, energies=energies, path=path_to_save_results + '/')

