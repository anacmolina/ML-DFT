#import warnings
#warnings.filterwarnings("ignore")

import os
import argparse
import time

import torch
import numpy as np

from flonacomldft.utils.io_utils import (
    load_pickle_file,
    load_csv_file,
    save_pickle_file, 
    get_project_path,
    save_json_args
)

from flonacomldft.sampling import run_metropolis
from flonacomldft.models.mixture import Mixture
from flonacomldft.internal_coordinates import Coordinates_mapping

#from flonacomldft.parallel import set_seed
from flonacomldft.utils.io_utils import get_process_id

#import gpaw.mpi as mpi

num_seed = [42] #set_seed()
torch.manual_seed(num_seed[0])
#mpi.world.barrier()

# for naming files
date_start = time.strftime('%Y-%m-%d %H:%M:%S')
process_id = get_process_id(date_start)

print('seed: ', num_seed)

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-np', '--num-procs', type=int, default=1)
parser.add_argument('-ml', '--mode-label', type=int, nargs='+', default=0)
parser.add_argument('-ids', '--ids', type=int, nargs='+', default=[None, None])
parser.add_argument('-rs', '--random-seed', type=str, default=str(num_seed))
parser.add_argument('-path', '--folder-path', type=str, default='database/')
parser.add_argument('-pid', '--process-id', type=int, default=int(process_id))
parser.add_argument('-ncs', '--n-chains-steps', type=int, nargs='+', default=[50, 100] )
parser.add_argument('-etype', '--energy-type', type=str, default='mlp')

args = parser.parse_args()

args.date_start = date_start

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#torch.set_num_threads(args.num_procs)
print('n_thread_set: ', args.num_procs)


print(args)

mode_labels = args.mode_label
ids = args.ids

# mcmc chains parameters

n_chains,n_steps = args.n_chains_steps
energy_type = args.energy_type

coord_mapping = Coordinates_mapping()

# load data

zmats_test = [load_csv_file(args.folder_path + "is{:d}_md_test.csv".format(mode_label)) for mode_label in mode_labels] 

xs = [coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 12]
                                    ) for mode_label, zmat_test in zip(mode_labels, zmats_test)]

xs = torch.stack([torch.cat((x[0], x[1].reshape(-1, 1), x[2].reshape(-1, 1), zmat_test[:, 14].reshape(-1, 1)), dim=1) for x, zmat_test in zip(xs, zmats_test)])
xs = xs.flatten(start_dim=0, end_dim=1).to(torch.float32)

# configs to initialize the chains

xs = xs[torch.randperm(xs.size()[0])]
xs = xs[:n_chains]

# mlp models

mlps_dic = [load_pickle_file(args.folder_path + 'is{:d}_mlp_dic_training_{:d}.pkl'.format(mode_label, id_)) for mode_label, id_ in zip(mode_labels, ids[:len(mode_labels)]) ]
mlp_models = np.array([mlp_dic['model'] for mlp_dic in mlps_dic])

print('# models: ', len(mlp_models))

# flow models

flows_dic = [load_pickle_file(args.folder_path + 'is{:d}_flow_dic_training_{:d}.pkl'.format(mode_label, id_)) for mode_label, id_ in zip(mode_labels, ids[len(mode_labels):]) ]
flow_models = np.array([flow_dic['model'] for flow_dic in flows_dic])

if len(mode_labels)==1:
    mixture = False
    flow_model = flow_models[0]
    mlp_models = mlp_models[0]
else:
    flow_model = Mixture(flow_models, torch.tensor([0.5, 0.5]).detach())
    mixture = True

# initialize metropolis simulation
# run mcmc

out = run_metropolis(
    model=flow_model,
    init=xs,
    n_chains=n_chains,
    n_steps=n_steps,
    name_run="", # TODO: number of runs
    energy_type=energy_type,
    frac_dft=0.2,
    mlp_models=mlp_models,
    mixture=mixture,
    T=300,
    with_tqdm=True,
)

date_end = time.strftime('%Y-%m-%d %H:%M:%S')
args.date_end = date_end

args.algorithm = 'metropolis.py'

argparse_dict = vars(args)
out['args'] = argparse_dict

save_json_args(args, 'metropolis', args.process_id, os.getcwd() + '/')

if mixture:
    f = "mixture_mcmc_chains_{:d}.pkl".format(args.process_id)
else:
    f = "is{:d}_mcmc_chains_{:d}.pkl".format(mode_labels[0], args.process_id)

save_pickle_file(out, f, path = os.getcwd() + '/')

import matplotlib.pyplot as plt

accs = out['accs'].mean(dim=1)

plt.figure()
plt.plot(accs)
plt.xlabel('steps')
plt.ylabel('acceptance rate')
plt.savefig('acceptance_rate_{:d}.png'.format(args.process_id))
