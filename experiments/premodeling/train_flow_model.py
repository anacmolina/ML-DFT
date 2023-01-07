import numpy as np
import torch

from flonacomldft.utils.io_utils import load_csv_file
from flonacomldft.utils.data_processing import (
    split_data_from_dataframe,
    centering_in_radian
)

from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow

from flonacomldft.collective_variables import get_CVs
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for flows

n_iter = 50
lr = 1e-3
mode_label = 2 # or 2

torch.manual_seed(100)

# load data

xs = load_csv_file("is{:d}_lcao_zmat.csv".format(mode_label))
xs = xs[:, :-2] # remove energy and logdetjac values

train_size = 0.8
sk_seed = 42

xs_train, xs_test = split_data_from_dataframe(xs, train_size, sk_seed)

xs_train_mean = xs_train.mean(dim=0)

xs_train, centering_args = centering_in_radian(xs_train)
xs_test = centering_in_radian(xs_train, xs_train_mean, return_centering_args=False)

model = RealNVP_MLP(12,
                    n_blocks=3,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    centering_args=centering_args,
                    device=device,
                    )

out = train_flow(
    model,
    xs_train,
    xs_test,
    n_iter=n_iter,
    lr=lr,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
)

x_sample = model.sample(100)

from flonacomldft.plots import plotting_fes_db

fig, ax = plotting_fes_db()
x_sample_cv = np.array(get_CVs(x_sample)).T
ax.scatter(x_sample_cv[:, 0], x_sample_cv[:, 1], label="mode {:d} - realnvp init".format(mode_label), c='C{:d}'.format(mode_label), alpha=0.5)
x_rad_cv = np.array(get_CVs(xs[:100])).T
ax.scatter(x_rad_cv[:, 0], x_rad_cv[:, 1], marker='x', c='C{:d}'.format(mode_label), label="mode {:d} - data".format(mode_label))
ax.legend()
plt.show()

from flonacomldft.utils.io_utils import get_project_path
from flonacomldft.utils.io_utils import save_pickle_file

f = get_project_path() + "database/is{:d}_flow_dic_training.pkl".format(mode_label)
save_pickle_file(out, f)

