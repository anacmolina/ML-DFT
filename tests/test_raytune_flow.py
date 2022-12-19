from ase.parallel import parprint as print

import time
import copy
import json

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flonacomldft.models.real_nvp import RealNVP_MLP, Angles_mapping
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.internal_coordinates import Structure, get_mix_data
from flonacomldft.utils.data_utils import get_path, save_pickle_file
from flonacomldft.collective_variables import get_CVs

from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.manual_seed(100)

def train_flow_is1(config):
    mode_label = 1
    df = pd.read_csv(get_path() + "is{:d}_lcao_zmat.csv".format(mode_label))
    U = df.energies
    X = df.drop(["energies"], axis=1)
    n = -1
    x_rad = torch.from_numpy(X[:n].to_numpy()).float()
    U_rad = torch.from_numpy(U[:n].to_numpy()).float()
    x_rad_center = x_rad.mean(dim=0)
    x_rad_centered = x_rad - x_rad_center

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
    n_iter = 10
    out = train_flow(
        model,
        x_real_centered,
        n_iter=n_iter,
        lr=config['lr'],
        # bs=100,
        use_scheduler=False,
        step_schedule=100,
        save_splits=10,
        grad_clip=1e4,
        use_tune=True,
    )
    # tune.report({"coin" :out['losses'][-1]})

search_space = {"lr": tune.grid_search([1e-3, 1e-4, 1e-5])}
tuner = tune.Tuner(
    train_flow_is1,
    param_space=search_space,
)
results = tuner.fit()
dfs = {result.log_dir: result.metrics_dataframe for result in results}
plt.figure(figsize=(15, 5)) 
axs = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
for d in dfs.values():
    d['_metric/loss'].plot(ax=axs[0], legend=False)
    d['_metric/grad_norm'].plot(ax=axs[1], legend=False)

plt.show(block=False)

# x_sample = model.sample(100)
# x_sample_cv = np.array(get_CVs(x_sample)).T
# plt.scatter(x_sample_cv[:, 0], x_sample_cv[:, 1], label="mode {:d} - realnvp init".format(mode_label), c='C{:d}'.format(mode_label), alpha=0.5)
# x_rad_cv = np.array(get_CVs(x_rad[:100])).T
# plt.scatter(x_rad_cv[:, 0], x_rad_cv[:, 1], marker='x', c='C{:d}'.format(mode_label), label="mode {:d} - data".format(mode_label))
# plt.legend()
# plt.show(block=False)

# from flonacomldft.utils.data_utils import get_project_path
# from flonacomldft.utils.data_utils import save_pickle_file

# f = get_project_path() + "tests/is{:d}_flow_dic_training.pkl".format(mode_label)
# save_pickle_file(out, f)