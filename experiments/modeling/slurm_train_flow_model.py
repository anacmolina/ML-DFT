import argparse
import numpy as np
import torch
import time

from flonacomldft.utils.io_utils import load_csv_file, save_pickle_file, get_path
from flonacomldft.internal_coordinates import Coordinates_mapping, join_data
from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow


# for naming files
date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)

print(get_path())

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-ni', '--n-iter', type=int, default=10)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-hdm', '--hidden-dim', type=int, default=64)
parser.add_argument('-hdp', '--hidden-depth', type=int, default=3)
parser.add_argument('-id', '--slurm-id', type=str, default=str(random_id))

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(100)

# load data
mode_label = args.mode_label #or 1 
zmat_train = load_csv_file("datasets/is{:d}_md_train.csv".format(mode_label))
zmat_test = load_csv_file("datasets/is{:d}_md_test.csv".format(mode_label))

# real centered frame
coord_mapping = Coordinates_mapping()
xs_train, logdetjacs_train, energies_train = coord_mapping.get_real_centered_from_internal(
                                    zmat_train[:, :12],
                                    zmat_train[:, 12],
                                    isomer=mode_label,
                                    energies=zmat_train[:, 13]
                                    )
xs_train = xs_train.to(torch.float32)


xs_test, logdetjacs_test, energies_test = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 12],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 13]
                                    )
xs_test = xs_test.to(torch.float32)

train = join_data(xs_train,
                logdetjacs_train,
                energies_train,
                zmat_train[:, 14])

test = join_data(xs_test,
                logdetjacs_test,
                energies_test,
                zmat_test[:, 14])

#train = torch.cat((xs_train, logdetjacs_train.reshape(-1, 1), energies_train.reshape(-1, 1), zmat_train[:, 14].reshape(-1, 1)), dim=1).to(torch.float32)
#test = torch.cat((xs_test, logdetjacs_test.reshape(-1, 1), energies_test.reshape(-1, 1), zmat_test[:, 14].reshape(-1, 1)), dim=1).to(torch.float32)

model = RealNVP_MLP(12,
                    n_blocks=args.n_blocks,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    base_cov=torch.cov(xs_train.T),
                    hidden_dim=args.hidden_dim,
                    hidden_depth=args.hidden_depth,
                    device=device,
                    )

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
)

# import numpy as np
# from flonacomldft.collective_variables import get_CVs 
# from flonacomldft.utils.plots import plotting_fes_db, plot_losses
# import matplotlib.pyplot as plt
# plot_losses(out['losses'][0], out['losses'][1])
# plt.show()
# xs_sample = out['model'].sample(100)
# coord_mapping = Coordinates_mapping()
# zs_sample, logdetjac_sample = coord_mapping.get_internal_from_real_centered(xs_sample, isomer=mode_label)
# from ase.visualize import view
# view(coord_mapping.build_molecule_from_zmat(zs_sample[0])
# )
# x_sample_cv = np.array(get_CVs(zs_sample)).T
# x_cv = np.array(get_CVs(zmat_train[:50, :12])).T
# fig, ax = plotting_fes_db()
# ax.scatter(x_sample_cv[:, 0], x_sample_cv[:, 1], label="mode {:d} - realnvp init".format(mode_label), c='C{:d}'.format(mode_label))#, alpha=0.5)
# ax.scatter(x_cv[:, 0], x_cv[:, 1], marker='x', c='C{:d}'.format(mode_label), label="mode {:d} - data".format(mode_label), alpha=0.5)
# ax.legend()
# plt.show()

# filename could also include the hyperparameters
f = "models/is{:d}_flow_dic_training_{:s}.pkl".format(mode_label, args.slurm_id)
save_pickle_file(out, f)

