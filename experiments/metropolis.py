import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import time

import torch
import numpy as np

from ase.parallel import parprint as print

from flonacomldft.utils.io_utils import (
    load_pickle_file,
    load_csv_file,
    save_pickle_file, 
    get_project_path,
    save_json_args
)

import gpaw.mpi as mpi

ranks = mpi.world.size
print('procs: ', ranks)

from flonacomldft.sampling import run_metropolis
from flonacomldft.models.mixture import Mixture
from flonacomldft.internal_coordinates import Coordinates_mapping

#from flonacomldft.parallel import set_seed
from flonacomldft.utils.io_utils import get_process_id


num_seed = [42] #set_seed()
torch.manual_seed(num_seed[0])
#mpi.world.barrier()

# for naming files
date_start = time.strftime('%Y-%m-%d %H:%M:%S')
process_id = get_process_id(date_start)

print('seed: ', num_seed)

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-np', '--num-procs', type=int, default=ranks)
parser.add_argument('-isomer', '--mode-label', type=int, nargs='+', default=[0])
parser.add_argument('-ids', '--ids', type=int, nargs='+', default=[None]) # Ideally this should load not be here, but its for identifying flow models
parser.add_argument('-rs', '--random-seed', type=str, default=str(num_seed))
parser.add_argument('-path', '--folder-path', type=str, default='database/')
parser.add_argument('-pid', '--process-id', type=int, default=int(process_id))
parser.add_argument('-nchains', '--n-chains', type=int, default=5)
parser.add_argument('-nsteps', '--n-steps', type=int, default=10)
parser.add_argument('-etype', '--energy-type', type=str, default='dft')

args = parser.parse_args()

args.date_start = date_start

print('args: ', args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#torch.set_num_threads(args.num_procs)
print('n_thread_set: ', args.num_procs)


print(args)

mode_labels = args.mode_label
ids = args.ids

# mcmc chains parameters

n_chains = args.n_chains
n_steps = args.n_steps
energy_type = args.energy_type

coord_mapping = Coordinates_mapping()

# load data

zmats_test = [load_csv_file(args.folder_path + "converged/is{:d}_flow_test.csv".format(mode_label)) for mode_label in mode_labels] 
print('zmats_test: ', zmats_test)
xs = [coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 12]
                                    ) for mode_label, zmat_test in zip(mode_labels, zmats_test)]
print('xs: ', xs)
xs = torch.stack([torch.cat((x[0], x[2].reshape(-1, 1), zmat_test[:, 13].reshape(-1, 1), x[1].reshape(-1, 1)), dim=1) for x, zmat_test in zip(xs, zmats_test)])
xs = xs.flatten(start_dim=0, end_dim=1).to(torch.float32)

print('xs together: ', xs)
# configs to initialize the chains

xs = xs[torch.randperm(xs.size()[0])]
xs = xs[:n_chains]

if args.energy_type == 'mlp':

    # mlp models

    mlps_dic = [load_pickle_file(args.folder_path + 'models/is{:d}_mlp_dic_training_{:d}.pkl'.format(mode_label, id_)) for mode_label, id_ in zip(mode_labels, ids[:len(mode_labels)]) ]
    mlp_models = np.array([mlp_dic['model'] for mlp_dic in mlps_dic])
    print('# models: ', len(mlp_models))

else:

    mlp_models = None


# flow models

print('load flow models: ', mode_labels, ids, len(mode_labels) )

flows_dic = [load_pickle_file(args.folder_path + 'models/is{:d}_flow_dic_training_{:d}.pkl'.format(mode_label, id_)) for mode_label, id_ in zip(mode_labels, ids)]
flow_models = np.array([flow_dic['model'] for flow_dic in flows_dic])

print('# flow models: ', len(flows_dic))

if len(mode_labels)==1:
    mixture = False
    flow_model = flow_models[0]

    if "mlp" in energy_type: 
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
    id_run=0, # TODO: number of runs
    energy_type=energy_type,
    frac_dft=0.2,
    mlp_models=mlp_models,
    mixture=mixture,
    T=300,
    with_tqdm=False,
    return_ratio = False,
    return_proposals = False,
    dft_folder_name="DFTComputations_{:d}".format(args.process_id),
    scheduler=1,
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
