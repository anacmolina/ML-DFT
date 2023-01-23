# from ase.parallel import parprint as print

# import time
# import copy
# import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flonacomldft.models.real_nvp import RealNVP_MLP, Angles_mapping
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.utils.io_utils import get_path, save_pickle_file
from flonacomldft.collective_variables import get_CVs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_iter_ = 50
lr_ = 1e-3
mode_label = 1 # or 2

torch.manual_seed(100)

df = pd.read_csv(get_path() + "is{:d}_lcao_zmat.csv".format(mode_label))
U = df.energies
X = df.drop(["energies"], axis=1)
n = -1
x_rad = torch.from_numpy(X[:n].to_numpy()).float()
U_rad = torch.from_numpy(U[:n].to_numpy()).float()

# centering in radian
x_rad_center = x_rad.mean(dim=0)
x_rad_centered = x_rad - x_rad_center

# computing the tanh in order to estimate the covariance for the flow base distribution 
x_real_centered, _ = Angles_mapping().rads_to_reals(x_rad_centered)
cov_real = torch.cov(x_real_centered.T)

centering_args = {"cov_base": cov_real, "mean_out": x_rad_center}
model = RealNVP_MLP(12,
                    n_blocks=3,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    centering_args=centering_args,
                    device=device,
                    )

out = train_flow(
    model,
    x_real_centered,
    x_real_centered, # passing centered values to the flows 
    ## to add x_test
    n_iter=n_iter_,
    lr=lr_,
    # bs=100,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
)

x_sample = model.sample(100)

from flonacomldft.utils.plots import plotting_fes_db

fig, ax = plotting_fes_db()
x_sample_cv = np.array(get_CVs(x_sample)).T
ax.scatter(x_sample_cv[:, 0], x_sample_cv[:, 1], label="mode {:d} - realnvp init".format(mode_label), c='C{:d}'.format(mode_label), alpha=0.5)
x_rad_cv = np.array(get_CVs(x_rad[:100])).T
ax.scatter(x_rad_cv[:, 0], x_rad_cv[:, 1], marker='x', c='C{:d}'.format(mode_label), label="mode {:d} - data".format(mode_label))
ax.legend()
plt.show()

from flonacomldft.utils.io_utils import get_project_path
from flonacomldft.utils.io_utils import save_pickle_file

#f = get_project_path() + "tests/is{:d}_flow_dic_training.pkl".format(mode_label)
f = "is{:d}_flow_dic_training.pkl".format(mode_label)
save_pickle_file(out, f)