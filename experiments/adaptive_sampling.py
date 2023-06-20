import warnings
warnings.filterwarnings("ignore")

# standard library imports
import os
import time
import argparse

# scientific library imports
import torch
import numpy as np

# parallelization set up
import gpaw.mpi as mpi
#from ase.parallel import parprint as print

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
# sampling and training
from flonacomldft.full_adaptive_sampling import adaptive_sampling

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

parser = argparse.ArgumentParser(description='Prepare experiment')
# execution params
parser.add_argument('-np', '--num-procs', type=int, default=(len(ranks)))
parser.add_argument('-pid', '--process-id', type=int, default=date_start)
parser.add_argument('-rs', '--random-seed', type=int, default=num_seed)
parser.add_argument('-path', '--folder-path', type=str, default='berendsen/')
# training params
parser.add_argument('-isomer', '--mode-label', type=int, nargs='+', default=[0])
parser.add_argument('-ids', '--ids', type=int, nargs='+', default=[None, None])
parser.add_argument('-ni', '--n-iter', type=int, default=1000)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
# adaptive sampling params
parser.add_argument('-nruns', '--n-runs', type=int, default=5)
parser.add_argument('-nchains', '--n-chains', type=int, default=5)
parser.add_argument('-nsteps', '--n-steps', type=int, default=10)
parser.add_argument('-etype', '--energy-type', type=str, default='dft')

args = parser.parse_args()
args.date_start = str(date_start)

# torch settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if len(ranks) > 1:
    torch.set_num_threads(int(args.num_procs))
    torch.manual_seed(args.random_seed)

print(torch.get_num_threads(), args.num_procs, rank)

mpi.world.barrier()

# isomer labels
mode_labels = args.mode_label
ids = args.ids

# mcmc chains parameters
n_runs = args.n_runs
n_chains = args.n_chains
n_steps = args.n_steps
energy_type = args.energy_type

path_datasets = get_path() + '/' + args.folder_path

# real center coordinates

coord_mapping = Coordinates_mapping()

# TODO: is this the best way to get the dataset?
# TODO: real center data in a file already, add collective varibles to csv file 
def get_dataset(name, path, mode_labels):
    
    zmats = [load_csv_file("converged/is{:d}_{:s}.csv".format(mode_label, name), path) for mode_label in mode_labels] 
    xs = [coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 12]
                                    ) for mode_label, zmat_test in zip(mode_labels, zmats)]

    xs = torch.stack([torch.cat((x[0], x[2].reshape(-1, 1), 
                                 zmat[:, 13].reshape(-1, 1), 
                                 #x[1].reshape(-1, 1),
                                 ), dim=1) for x, zmat in zip(xs, zmats)])
    xs = xs.flatten(start_dim=0, end_dim=1).to(torch.float32)

    return xs, zmats

dataset_labels = ['flow_train', 'flow_test']

flow_xs_train, flow_zmat_train = get_dataset('flow_train', path_datasets, mode_labels)
flow_xs_test, flow_zmat_test = get_dataset('flow_test', path_datasets, mode_labels)

# flow models
flows_dic = [load_pickle_file("models/is{:d}_flow_dic_training_{:d}.pkl".format(mode_label, id_), path_datasets) for mode_label, id_ in zip(mode_labels, ids) ]

mpi.world.barrier()

# whether to use a mixture of flows
if len(mode_labels)==1:
    mixture = False
    flow_model = flows_dic[0]

    if "mlp" in energy_type: 
        ## ADD MLP MODEL
        pass
        #mlp_models = mlp_models[0]
else:
    mixture = True

if mixture:
    simulation_name = "mixture"
else:
    simulation_name = "is{:d}".format(mode_labels[0])

# path to save results
folder_to_save_results = 'results_adaptive_{:s}_{:d}'.format(simulation_name, args.process_id)
path_to_save_results = os.getcwd() + '/' + folder_to_save_results
if rank == 0:
    if not os.path.exists(path_to_save_results):
        os.makedirs(path_to_save_results)

mpi.world.barrier()


# retraining hyperparameters
flow_hyperparams = {'n_iter': args.n_iter,
    'lr': args.learning_rate,
    'use_scheduler': False,
    'step_schedule': 100,
    'save_splits': 5,
    'grad_clip': 1e4,
    'compute_part_ratio': True,
    }

# if mlp models are used
if "mlp" in energy_type:

    dataset_labels = dataset_labels + ['mlp_train', 'mlp_test']

    mlp_xs_train, mlp_zmat_train = get_dataset('mlp_train', args.folder_path, mode_labels)
    mlp_xs_test, mlp_zmat_test = get_dataset('mlp_test', args.folder_path, mode_labels)

    mlps_dic = [load_pickle_file(args.folder_path + 'models/is{:d}_mlp_dic_training_{:d}.pkl'.format(mode_label, id_)) for mode_label, id_ in zip(mode_labels, ids[:len(mode_labels)]) ]

    print('# models: ', len(mlps_dic))

    mlp_hyperparams = {'n_iter': args.n_iter,
        'lr': args.learning_rate,
        'use_scheduler': False,
        'step_schedule': 100,
    }
else:
    mlp_hyperparams = None
    mlps_dic = None

    mlp_xs_train = None
    mlp_xs_test = None

    mlp_zmat_train = None
    mlp_zmat_test = None

# init chains
shuffle = torch.randperm(flow_xs_test.shape[0])
init_mcmc = flow_xs_test[shuffle].clone()[:n_chains]

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
    )

date_end = time.strftime('%Y-%m-%d %H:%M:%S')
args.date_end = date_end

args.algorithm = 'adaptive_sampling.py'

argparse_dict = vars(args)
out['args'] = argparse_dict

save_json_args(args, 'adaptive_sampling', args.process_id, path_to_save_results)

f = "adaptative_sampling_{:d}.pkl".format(args.process_id)
save_pickle_file(out, f, path = path_to_save_results)
