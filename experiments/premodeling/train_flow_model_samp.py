import torch

from flonacomldft.utils.data_processing import centering_in_radian
from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.utils.io_utils import (
    load_csv_file,
    save_pickle_file
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for flows

n_iter = 10000
lr = 1e-4
mode_label = 2 # or 2

torch.manual_seed(100)

# load data

xs_train = load_csv_file("datasets/is{:d}_md_train.csv".format(mode_label))[:, :12]
xs_test = load_csv_file("datasets/is{:d}_md_test.csv".format(mode_label))[:, :12] # remove energy and logdetjac values
us_test = load_csv_file("datasets/is{:d}_md_test.csv".format(mode_label))[:, 13] 

xs_train_mean = xs_train.mean(dim=0)

xs_rad_train, centering_args = centering_in_radian(xs_train)
xs_rad_test = centering_in_radian(xs_test, xs_train_mean, return_centering_args=False)

model = RealNVP_MLP(12,
                    n_blocks=12,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    centering_args=centering_args,
                    hidden_dim=32,  #128 #32
                    hidden_depth=8,  #4   #8
                    device=device,
                    )

out = train_flow(
    model,
    xs_rad_train,
    xs_rad_test,
    us_test,
    isomer=mode_label,
    n_iter=n_iter,
    lr=lr,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
)

import numpy as np
from flonacomldft.collective_variables import get_CVs 
from flonacomldft.utils.plots import plotting_fes_db, plot_losses
import matplotlib.pyplot as plt
plot_losses(out['losses'][0], out['losses'][1])
plt.show()
xs_sample = out['model'].sample(100)
x_sample_cv = np.array(get_CVs(xs_sample)).T
x_cv = np.array(get_CVs(xs_train[:50])).T

fig, ax = plotting_fes_db()
ax.scatter(x_sample_cv[:, 0], x_sample_cv[:, 1], label="mode {:d} - realnvp init".format(mode_label), c='C{:d}'.format(mode_label))#, alpha=0.5)
ax.scatter(x_cv[:, 0], x_cv[:, 1], marker='x', c='C{:d}'.format(mode_label), label="mode {:d} - data".format(mode_label), alpha=0.5)
ax.legend()
plt.show()

#f = "models/is{:d}_flow_dic_training.pkl".format(mode_label)
#save_pickle_file(out, f)