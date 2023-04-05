import os
import argparse
import time

import torch
import numpy as np

from flonacomldft.utils.io_utils import (
    load_csv_file,
    load_pickle_file,
    save_pickle_file,
    get_project_path,
    save_json_args
)

from flonacomldft.internal_coordinates import Coordinates_mapping, join_data
from flonacomldft.full_adaptive_sampling import adaptative_sampling

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
parser.add_argument('-ni', '--n-iter', type=int, default=10)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-ml', '--mode-label', type=int, nargs='+', default=0)
parser.add_argument('-ids', '--ids', type=int, nargs='+', default=[None, None])
parser.add_argument('-rs', '--random-seed', type=str, default=str(num_seed))
parser.add_argument('-path', '--folder-path', type=str, default='database/')
parser.add_argument('-pid', '--process-id', type=int, default=int(process_id))
parser.add_argument('-nruns', '--n-runs', type=int, default=20)
parser.add_argument('-ncs', '--n-chains-steps', type=int, nargs='+', default=[50, 20] )
parser.add_argument('-etype', '--energy-type', type=str, default='mlp')

args = parser.parse_args()

args.date_start = date_start

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#torch.set_num_threads(args.num_procs)
print('n_thread_set: ', args.num_procs)

# TODO: Check the mlp is1 training loss

energy_type = args.energy_type

# mcmc params
n_runs = args.n_runs
n_chains, n_steps = args.n_chains_steps

#--------------------------------------------------

flow_hyperparams_is0 = {'n_iter': args.n_iter,
    'lr': args.learning_rate,
    'use_scheduler': False,
    'step_schedule': 100,
    'save_splits': 10,
    'grad_clip': 1e4}

flow_hyperparams_is1 = {'n_iter': args.n_iter,
    'lr': args.learning_rate,
    'use_scheduler': False,
    'step_schedule': 100,
    'save_splits': 10,
    'grad_clip': 1e4}

mlp_hyperparams_is0 = {'n_iter': args.n_iter,
    'lr': args.learning_rate,
    'use_scheduler': False,
    'step_schedule': 100,
}

mlp_hyperparams_is1 = {'n_iter': args.n_iter,
    'lr': args.learning_rate,
    'use_scheduler': False,
    'step_schedule': 100,
}
#--------------------------------------------------

mode_labels = args.mode_label
ids = args.ids

dataset_labels = ['md_train', 'md_test', 'flow_train', 'flow_test']

coord_mapping = Coordinates_mapping()

def get_dataset(name, path, mode_labels):
    
    zmats = [load_csv_file(args.folder_path + "is{:d}_{:s}.csv".format(mode_label, name)) for mode_label in mode_labels] 
    xs = [coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 12]
                                    ) for mode_label, zmat_test in zip(mode_labels, zmats)]

    xs = torch.stack([torch.cat((x[0], x[1].reshape(-1, 1), x[2].reshape(-1, 1), zmat[:, 14].reshape(-1, 1)), dim=1) for x, zmat in zip(xs, zmats)])
    xs = xs.flatten(start_dim=0, end_dim=1).to(torch.float32)

    return xs, zmats

flow_xs_train, flow_zmat_train = get_dataset('md_train', args.folder_path, mode_labels)
flow_xs_test, flow_zmat_test = get_dataset('md_test', args.folder_path, mode_labels)

mlp_xs_train, mlp_zmat_train = get_dataset('mlp_train', args.folder_path, mode_labels)
mlp_xs_test, mlp_zmat_test = get_dataset('mlp_test', args.folder_path, mode_labels)

# pretrain flows and mlps

# mlp models

mlps_dic = [load_pickle_file(args.folder_path + 'is{:d}_mlp_dic_training_{:d}.pkl'.format(mode_label, id_)) for mode_label, id_ in zip(mode_labels, ids[:len(mode_labels)]) ]
mlp_models = np.array([mlp_dic['model'] for mlp_dic in mlps_dic])

print('# models: ', len(mlp_models))

# flow models

flows_dic = [load_pickle_file(args.folder_path + 'is{:d}_flow_dic_training_{:d}.pkl'.format(mode_label, id_)) for mode_label, id_ in zip(mode_labels, ids[len(mode_labels):]) ]
flow_models = np.array([flow_dic['model'] for flow_dic in flows_dic])

# run adaptive sampling

out = adaptative_sampling(
    flow_init_train=flow_xs_train[0],
    flow_init_test=flow_xs_test[1],
    n_runs=n_runs,
    n_chains=n_chains,
    n_steps=n_steps,
    energy_type=energy_type,
    dict_flows_init=flows_dic,
    flow_hyperparams=[flow_hyperparams_is0, flow_hyperparams_is1],
    retraining_mlp=True,
    dict_mlps_init=mlps_dic,
    mlp_hyperparams=[mlp_hyperparams_is0, mlp_hyperparams_is1],
    mlp_init_train=mlp_xs_train,
    mlp_init_test=mlp_xs_test,
)

date_end = time.strftime('%Y-%m-%d %H:%M:%S')
args.date_end = date_end

args.algorithm = 'full_adaptive_sampling.py'

argparse_dict = vars(args)
out['args'] = argparse_dict

save_json_args(args, 'adaptive_sampling', args.process_id, os.getcwd() + '/')

f = "adaptative_sampling_{:d}.pkl".format(args.process_id)
save_pickle_file(out, f, path = os.getcwd() + '/')
