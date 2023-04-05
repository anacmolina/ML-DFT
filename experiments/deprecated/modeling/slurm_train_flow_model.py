import argparse
import numpy as np
import torch
import time

from flonacomldft.utils.io_utils import load_csv_file, save_pickle_file, get_path
from flonacomldft.internal_coordinates import Coordinates_mapping, join_data
from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow


# for naming files
date_start = time.strftime('%H:%M:%S %d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)

print(get_path())

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-np', '--num-procs', type=int, default=1)
parser.add_argument('-ni', '--n-iter', type=int, default=10)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-hdm', '--hidden-dim', type=int, default=64)
parser.add_argument('-hdp', '--hidden-depth', type=int, default=3)
parser.add_argument('-id', '--slurm-id', type=str, default=str(random_id))

args = parser.parse_args()

args.date_start = date_start
args.random_seed = random_id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('n_thread_set: ', args.num_procs)

torch.set_num_threads(int(args.num_procs))

torch.manual_seed(100)

# load data
mode_label = args.mode_label #or 1 
zmat_train = load_csv_file("datasets/is{:d}_md_train.csv".format(mode_label))
zmat_test = load_csv_file("datasets/is{:d}_md_test.csv".format(mode_label))

# real centered frame
coord_mapping = Coordinates_mapping()
xs_train, logdetjacs_train, energies_train = coord_mapping.get_real_centered_from_internal(
                                    zmat_train[:, :12],
                                    zmat_train[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_train[:, 12]
                                    )
xs_train = xs_train.to(torch.float32)


xs_test, logdetjacs_test, energies_test = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 12]
                                    )
xs_test = xs_test.to(torch.float32)

train = join_data(xs_train,
                energies_train,
                zmat_train[:, 13],
                logdetjacs_train,
).detach()

test = join_data(xs_test,
                energies_test,
                zmat_test[:, 13],
                logdetjacs_test,
).detach()


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

#print(args.n_blocks)
#print(args.hidden_depth)
#print(args.hidden_dim)

print('n_blocks', model.n_blocks, type(model.n_blocks))
print('hidden_depth', model.n_layers_in_coupling, type(model.n_layers_in_coupling))
print('hidden_dim', model.hidden_dim_in_coupling, type(model.hidden_dim_in_coupling))

print('threads: {:d}'.format(torch.get_num_threads()))

#from flonacomldft.utils.io_utils import load_pickle_file 
#mlp = load_pickle_file("models/is{:d}_mlp_dic_training.pkl".format(mode_label))['model']


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
    compute_ratio_acc=True,
    #mlp_model=mlp,
    n_chains=5,
    with_tqdm=True
)

date_end = time.strftime('%H:%M:%S %d-%m-%Y')
args.date_end = date_end

argparse_dict = vars(args)
out['args'] = argparse_dict

import matplotlib.pyplot as plt
plt.plot(out['ratios'])
plt.show()

# filename could also include the hyperparameters
f = "models/flow_tracking/is{:d}_flow_dic_training_{:s}.pkl".format(mode_label, args.slurm_id)
save_pickle_file(out, f)

import json

filename_args = "args_" + args.slurm_id + ".json"

with open(get_path() + "models/flow_tracking/" + filename_args, "w") as outfile:
    json.dump(argparse_dict, outfile)