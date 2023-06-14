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
    set_str_date_to_int,
    load_csv_file,
    save_json_args,
    save_pickle_file
    )
# coordinates handling
from flonacomldft.internal_coordinates import (
    Coordinates_mapping,
    join_data,
    )
# nf training
from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow

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
parser.add_argument('-np', '--num-procs', type=int, default=len(ranks))
parser.add_argument('-pid', '--process-id', type=int, default=date_start)
parser.add_argument('-rs', '--random-seed', type=int, default=num_seed)
parser.add_argument('-path', '--folder-path', type=str, default='database/berendsen/converged/')
# training params
parser.add_argument('-isomer', '--mode-label', type=int, default=0)
parser.add_argument('-ni', '--n-iter', type=int, default=10)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-node', '--hidden-dim', type=int, default=64)
parser.add_argument('-layer', '--hidden-depth', type=int, default=3)
# sampling params
parser.add_argument('-r', '--do-ratios', type=bool, default=False)
parser.add_argument('-etype', '--energy-type', type=str, default='dft')
parser.add_argument('-prop', '--n-prop', type=int, default=5)

args = parser.parse_args()
args.date_start = str(date_start)

print('args: ', args, rank)

# torch settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if len(ranks) > 1:
    torch.set_num_threads(int(args.num_procs))
    torch.manual_seed(args.random_seed)

# internal coordinates datasets
zmat_train = load_csv_file(args.folder_path + "is{:d}_flow_train.csv".format(args.mode_label))
zmat_test = load_csv_file(args.folder_path + "is{:d}_flow_test.csv".format(args.mode_label))

# real center coordinates datasets
coord_mapping = Coordinates_mapping()
xs_train, logdetjacs_train, energies_train = coord_mapping.get_real_centered_from_internal(
                                    zmat_train[:, :12],
                                    zmat_train[:, 14],
                                    isomer=args.mode_label,
                                    energies=zmat_train[:, 12]
                                    )
xs_train = torch.tensor(xs_train, dtype=torch.float32)

xs_test, logdetjacs_test, energies_test = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=args.mode_label,
                                    energies=zmat_test[:, 12]
                                    )
xs_test = torch.tensor(xs_test, dtype=torch.float32)

# join real center coordinates, energies and logdetjacs in a single tensor
train = join_data(xs_train, energies_train, zmat_train[:, xs_train.shape[1]+1], logdetjacs_train).detach()
test = join_data(xs_test, energies_test, zmat_test[:, xs_test.shape[1]+1], logdetjacs_test).detach()

# init flow model

cov = torch.cov(xs_train.T).detach()
model = RealNVP_MLP(dim=xs_train.shape[1],
                    n_blocks=args.n_blocks,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    base_cov=cov,
                    hidden_dim=args.hidden_dim,
                    hidden_depth=args.hidden_depth,
                    device=device,
                    )

mpi.world.barrier()

# path to save results
path_to_save_results = 'results_is{:d}_flow_{:d}'.format(args.mode_label, args.process_id)
if rank == 0:
    if not os.path.exists(path_to_save_results):
        os.makedirs(path_to_save_results)

mpi.world.barrier()

# training flow model
flow_dic = train_flow(
    model,
    train,
    test,
    n_iter=args.n_iter,
    lr=args.learning_rate,
    use_scheduler=True,
    step_schedule=int(args.n_iter/5),
    save_splits=10,
    grad_clip=1e4,
    with_tqdm=False,
    compute_part_ratio=True,
    energy_type='dft',
    n_prop=args.n_prop,
    path=path_to_save_results,
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

args.algorithm = 'train_flow_model'

argparse_dic = vars(args)
flow_dic['args'] = argparse_dic

# save output to pickle file
save_pickle_file(flow_dic, "is{:d}_flow_dic_training_{:d}.pkl".format(args.mode_label, args.process_id), path=path_to_save_results+'/')

# save arguments to json file

save_json_args(args, 'train_flow_model', args.process_id, path=os.getcwd()+'/'+path_to_save_results+'/')

# plotting libraries
import matplotlib.pyplot as plt
from flonacomldft.collective_variables import get_CVs
from flonacomldft.utils.plots import Flonaco_Plotter

xs = flow_dic['model'].sample(150)
zmats = coord_mapping.get_internal_from_real_centered(xs, isomer=args.mode_label)[0]
cvs = np.array(get_CVs(zmats))

flow_plotter = Flonaco_Plotter()

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.set_title("ml: {:d}, nb: {:d}, hdm: {:d}, hdp: {:d}".format(args.mode_label, args.n_blocks, args.hidden_dim, args.hidden_depth))
flow_plotter.plot_losses(np.array(flow_dic['losses'])*(-1), yscale=True, ax=ax)
plt.savefig(path_to_save_results + '/is{:d}_flow_losses_{:d}.png'.format(args.mode_label, args.process_id))

flow_plotter.plot_collective_variables_on_fes(cvs.T, label='Flow samples is{:d}'.format(args.mode_label))
plt.title("ml: {:d}, nb: {:d}, hdm: {:d}, hdp: {:d}".format(args.mode_label, args.n_blocks, args.hidden_dim, args.hidden_depth))
plt.savefig(path_to_save_results + '/is{:d}_flow_fes_{:d}.png'.format(args.mode_label, args.process_id))

from flonacomldft.utils.plots import set_plot_iteration

fig, ax = plt.subplots(1,1, figsize=(6,4))
set_plot_iteration(torch.stack(flow_dic['part_ratios']).detach(), avg=True, window_size=5, axis=1, ax=ax, label='part_ratio')
ax.set_xlabel('Training steps')
ax.set_ylabel('Participation ratio')
ax.legend()
plt.savefig(path_to_save_results + '/is{:d}_flow_part_ratio_{:d}.png'.format(args.mode_label, args.process_id))




