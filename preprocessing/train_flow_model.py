import argparse

import time
import torch
import numpy as np

from flonacomldft.utils.io_utils import (
    load_csv_file, 
    save_pickle_file,
    get_project_path,
    save_json_args 
)

from flonacomldft.internal_coordinates import (
    Coordinates_mapping,
    join_data
)

from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.parallel import set_seed
from flonacomldft.utils.io_utils import get_process_id

import gpaw.mpi as mpi

num_seed = set_seed()
mpi.world.barrier()

# for naming files
date_start = time.strftime('%Y-%m-%d %H:%M:%S')
process_id = get_process_id(date_start)

print('seed: ', num_seed)

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-np', '--num-procs', type=int, default=1)
parser.add_argument('-ni', '--n-iter', type=int, default=10)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-hdm', '--hidden-dim', type=int, default=64)
parser.add_argument('-hdp', '--hidden-depth', type=int, default=3)
parser.add_argument('-id', '--id', type=int, default=22249210)
parser.add_argument('-rs', '--random-seed', type=str, default=str(num_seed))
parser.add_argument('-path', '--folder-path', type=str, default='database/')
parser.add_argument('-pid', '--process-id', type=int, default=int(process_id))
parser.add_argument('-ncs', '--n-chains-steps', type=int, nargs='+', default=[50, 100] )
parser.add_argument('-udft', '--use-dft', type=bool, default=False)

args = parser.parse_args()

args.date_start = date_start

print(args.n_chains_steps, args.n_chains_steps[0], args.n_chains_steps[1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('n_thread_set: ', args.num_procs)

torch.set_num_threads(int(args.num_procs))

# load data
mode_label = args.mode_label
zmat_train = load_csv_file(args.folder_path + "is{:d}_md_train.csv".format(mode_label))
zmat_test = load_csv_file(args.folder_path + "is{:d}_md_test.csv".format(mode_label))

# real centered frame
coord_mapping = Coordinates_mapping()
xs_train, logdetjacs_train, energies_train = coord_mapping.get_real_centered_from_internal(
                                    zmat_train[:, :12],
                                    zmat_train[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_train[:, 12]
                                    )
xs_train = torch.tensor(xs_train, dtype=torch.float32)

xs_test, logdetjacs_test, energies_test = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 12]
                                    )
xs_test = torch.tensor(xs_test, dtype=torch.float32)

# join data
train = join_data(xs_train, energies_train, zmat_train[:, 13], logdetjacs_train).detach()
test = join_data(xs_test, energies_test, zmat_test[:, 13], logdetjacs_test).detach()

cov = torch.cov(xs_train.T).detach()
model = RealNVP_MLP(12,
                    n_blocks=args.n_blocks,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    base_cov=cov,
                    hidden_dim=args.hidden_dim,
                    hidden_depth=args.hidden_depth,
                    device=device,
                    )

print('n_blocks', model.n_blocks, type(model.n_blocks))
print('hidden_depth', model.n_layers_in_coupling, type(model.n_layers_in_coupling))
print('hidden_dim', model.hidden_dim_in_coupling, type(model.hidden_dim_in_coupling))

#print('threads: {:d}'.format(torch.get_num_threads()))
                                
from flonacomldft.utils.io_utils import load_pickle_file 
mlp = load_pickle_file(args.folder_path + "is{:d}_mlp_dic_training_{:d}.pkl".format(mode_label, args.id))['model']

out = train_flow(
    model,
    train,
    test,
    n_iter=args.n_iter,
    lr=args.learning_rate,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
    use_dft=args.use_dft,
    compute_ratios=True,
    mlp_model=mlp,
    n_chains=args.n_chains_steps[0],
    n_steps=args.n_chains_steps[1],
    with_tqdm=False,
)

date_end = time.strftime('%Y-%m-%d %H:%M:%S')
args.date_end = date_end

args.algorithm = 'train_flow_model.py'

argparse_dict = vars(args)
out['args'] = argparse_dict

save_pickle_file(out, args.folder_path + "is{:d}_flow_dic_training_{:d}.pkl".format(mode_label, args.process_id))

save_json_args(args, 'train_flow_model', args.process_id, get_project_path() + args.folder_path)

import matplotlib.pyplot as plt
from flonacomldft.utils.plots import (
    plot_losses,
)

path = get_project_path() + args.folder_path

plot_losses(out['losses'][0], out['losses'][1], log_yscale=False)
plt.savefig(path + 'loss_flow_is{:d}_{:d}.png'.format(mode_label, args.process_id))

acc_ratios = [torch.stack(out['ratios'][i]['mlp']['acc_ratios']).mean(dim=1)[-1].detach().numpy() for i in range(10)]
part_ratios = [torch.stack(out['ratios'][i]['mlp']['part_ratios'])[-1].detach().numpy() for i in range(10)]
plt.figure()
plt.plot(acc_ratios, label='acceptance')
plt.plot(part_ratios, label='participation')
plt.legend()
plt.savefig(path + 'acc_part_ratios_{:d}.png'.format(args.process_id))