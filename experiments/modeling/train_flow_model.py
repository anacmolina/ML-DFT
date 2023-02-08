import torch

from flonacomldft.utils.io_utils import load_csv_file, save_pickle_file
from flonacomldft.internal_coordinates import Coordinates_mapping, join_data
from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for flows

n_iter = 10000
lr = 5e-5

torch.manual_seed(100)

# load data

mode_label = 1 #or 1 

zmat_train = load_csv_file("datasets/is{:d}_md_train.csv".format(mode_label))
zmat_test = load_csv_file("datasets/is{:d}_md_test.csv".format(mode_label))

# real centered frame

coord_mapping = Coordinates_mapping()
xs_train, logdetjacs_train, energies_train = coord_mapping.get_real_centered_from_internal(
                                    zmat_train[:, :12],
                                    isomer=mode_label,
                                    energies=zmat_train[:, 12],
                                    logdetjacs=zmat_train[:, 14],
                                    )
xs_train = xs_train.to(torch.float32)


xs_test, logdetjacs_test, energies_test = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 12],
                                    logdetjacs=zmat_test[:, 14],
                                    )
xs_test = xs_test.to(torch.float32)

train = join_data(xs_train,
                energies_train,
                zmat_train[:, 13],
                logdetjacs_train,
)

test = join_data(xs_test,
                energies_test,
                zmat_test[:, 13],
                logdetjacs_test,
)


model = RealNVP_MLP(12,
                    n_blocks=12,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    base_cov=torch.cov(xs_train.T).detach(),
                    hidden_dim=128,  #128 #32
                    hidden_depth=4,  #4   #8
                    device=device,
                    )

out = train_flow(
    model,
    train,
    test,
    n_iter=n_iter,
    lr=lr,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
)
"""
import numpy as np
from flonacomldft.collective_variables import get_CVs 
from flonacomldft.utils.plots import plotting_fes_db, plot_losses
import matplotlib.pyplot as plt
plot_losses(out['losses'][0], out['losses'][1])
plt.show()

xs_sample = out['model'].sample(100)

coord_mapping = Coordinates_mapping()

zs_sample, logdetjac_sample = coord_mapping.get_internal_from_real_centered(xs_sample, isomer=mode_label)

from ase.visualize import view
view(coord_mapping.build_molecule_from_zmat(zs_sample[0])
)

x_sample_cv = np.array(get_CVs(zs_sample)).T


x_cv = np.array(get_CVs(zmat_train[:50, :12])).T
fig, ax = plotting_fes_db()
ax.scatter(x_sample_cv[:, 0], x_sample_cv[:, 1], label="mode {:d} - realnvp init".format(mode_label), c='C{:d}'.format(mode_label))#, alpha=0.5)
ax.scatter(x_cv[:, 0], x_cv[:, 1], marker='x', c='C{:d}'.format(mode_label), label="mode {:d} - data".format(mode_label), alpha=0.5)
ax.legend()
plt.show()
"""

f = "models/is{:d}_flow_dic_training.pkl".format(mode_label)
save_pickle_file(out, f)

