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
from ase.parallel import parprint as print

# flonaco imports
# io handling
from flonacomldft.utils.io_utils import (
    load_pickle_file,
    load_csv_file,
    save_pickle_file, 
    save_json_args,
    set_str_date_to_int
)
# sampling 
from flonacomldft.sampling import run_metropolis
from flonacomldft.models.mixture import Mixture
from flonacomldft.internal_coordinates import Coordinates_mapping

ranks = np.arange(0, mpi.world.size)
rank = mpi.rank
comm = mpi.world.new_communicator(ranks)

num_seed = np.array([0])
date_start = np.array([0])

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

# define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
# execution params
parser.add_argument('-np', '--num-procs', type=int, default=len(ranks))
parser.add_argument('-pid', '--process-id', type=int, default=date_start)
parser.add_argument('-rs', '--random-seed', type=int, default=num_seed)
parser.add_argument('-path', '--folder-path', type=str, default='database/berendsen/')
# sampling params
parser.add_argument('-isomer', '--mode-label', type=int, nargs='+', default=[0])
parser.add_argument('-ids', '--ids', type=int, nargs='+', default=[None]) # Ideally this should load not be here, but its for identifying flow models
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

# isomer labels
mode_labels = args.mode_label
ids = args.ids

# mcmc chains parameters

n_chains = args.n_chains
n_steps = args.n_steps
energy_type = args.energy_type

# real center coordinates
coord_mapping = Coordinates_mapping()
zmats_test = [load_csv_file(args.folder_path + "converged/is{:d}_flow_test.csv".format(mode_label)) for mode_label in mode_labels] 

xs = [coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 12]
                                    ) for mode_label, zmat_test in zip(mode_labels, zmats_test)]

# join real center coordinates, energies, and logdetjacs in a single tensor
xs = torch.stack([torch.cat((x[0], x[2].reshape(-1, 1), zmat_test[:, 13].reshape(-1, 1), x[1].reshape(-1, 1)), dim=1) for x, zmat_test in zip(xs, zmats_test)])
xs = xs.flatten(start_dim=0, end_dim=1).to(torch.float32)

# configs to initialize the chains
xs = xs[torch.randperm(xs.size()[0])]
xs = xs[:n_chains]

# load flow models
flows_dic = [load_pickle_file(args.folder_path + 'models/is{:d}_flow_dic_training_{:d}.pkl'.format(mode_label, id_)) for mode_label, id_ in zip(mode_labels, ids[:len(mode_labels)])]
flow_models = np.array([flow_dic['model'] for flow_dic in flows_dic])

print('# flow models: ', len(flows_dic))

# load mlp models
if args.energy_type == 'mlp':
    mlps_dic = [load_pickle_file(args.folder_path + 'models/is{:d}_mlp_dic_training_{:d}.pkl'.format(mode_label, id_)) for mode_label, id_ in zip(mode_labels, ids[:len(mode_labels)]) ]
    mlp_models = np.array([mlp_dic['model'] for mlp_dic in mlps_dic])
    print('# mlp models: ', len(mlp_models))
else:
    mlp_models = None

# whether to use a mixture of flows
if len(mode_labels)==1:
    mixture = False
    flow_model = flow_models[0]

    if "mlp" in energy_type: 
        mlp_models = mlp_models[0]
else:
    flow_model = Mixture(flow_models, torch.tensor([0.5, 0.5]).detach())
    mixture = True

if mixture:
    simulation_name = "mixture"
else:
    simulation_name = "is{:d}".format(mode_labels[0])

# path to save results
path_to_save_results = 'results_{:s}_{:d}'.format(simulation_name, args.process_id)
if rank == 0:
    if not os.path.exists(path_to_save_results):
        os.makedirs(path_to_save_results)

mpi.world.barrier()

# initialize metropolis simulation
metropolis_dic = run_metropolis(
    model=flow_model,
    init=xs,
    n_chains=n_chains,
    n_steps=n_steps,
    id_run=0, # TODO: number of runs
    energy_type=energy_type,
    frac_dft=0.2,
    mlp_models=mlp_models,
    mixture=mixture,
    T=300,
    with_tqdm=False,
    return_ratio = False,
    return_proposals = False,
    dft_folder_name=path_to_save_results+'/DFTComputations_{:d}'.format(args.process_id),
    scheduler=10,
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
metropolis_dic['args'] = argparse_dic

# save output to pickle file
save_pickle_file(metropolis_dic, "{:s}_mcmc_dic_{:d}.pkl".format(simulation_name, args.process_id), path = os.getcwd() + '/' + path_to_save_results + '/')

# save output to json file
save_json_args(args, 'metropolis', args.process_id, os.getcwd() + '/' + path_to_save_results + '/')

import matplotlib.pyplot as plt
from flonacomldft.utils.plots import set_plot_iteration 
from flonacomldft.utils.plots import Flonaco_Plotter

accs = metropolis_dic['accs'].mean(dim=1)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
set_plot_iteration(accs.mean(dim=1), avg=True, window_size=10, axis=1, ax=ax)
ax.set_xlabel('Iteration')
ax.set_ylabel('Acceptance rate')
plt.savefig(path_to_save_results + '/acceptance_rate_{:d}.png'.format(args.process_id))

#xs = metropolis_dic['xs']
