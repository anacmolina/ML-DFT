import time
import copy

import torch
import numpy as np 
import pandas as pd 

from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.internal_coordinates import Angles_mapping

from flonacomldft.utils.data_utils import (
    get_path, 
    save_pickle_file,
    split_data_from_dataframe
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random_id = np.random.randint(100)
torch.manual_seed(random_id)

sk_seed = 42
train_size = 0.8

n_md = 2500 # 5000 steps in total
df_md = pd.read_csv(get_path() + 'is2_lcao_zmat.csv').loc[:n_md]

x_train_md, x_test_md, y_train_md, y_test_md = split_data_from_dataframe(df_md, train_size, sk_seed)

del y_test_md, y_train_md

Angles_mapping().inv_mapping(x_train_md)
Angles_mapping().inv_mapping(x_test_md)

cov = torch.cov(x_train_md.T)
mean = x_train_md.mean(0)

args_rnvp = {
    "dim": x_train_md.shape[1],
    "n_realnvp_block": 5,
    "block_depth": 1,
    "args_prior": {
        "type": "white",
        "cov": cov,
        "mean": mean,
    },  # Gaussian with non-trival mean and covariance for base
    "init_weight_scale": 1e-6,
}

model = RealNVP_MLP(
    args_rnvp["dim"],
    args_rnvp["n_realnvp_block"],
    args_rnvp["block_depth"],
    init_weight_scale=args_rnvp["init_weight_scale"],
    prior_arg=args_rnvp["args_prior"],
    device=device,
)

model_init = copy.deepcopy(model)

flow_hyperparams = {'n_iter': 1000,
    'lr': 5e-4,
    'use_scheduler': False,
    'step_schedule': 100,
    'save_splits': 10,
    'grad_clip': 1e4}

_ = train_flow(
    model,
    x_train_md,
    x_test_md,
    **flow_hyperparams,
)

save_pickle_file(_, "flow_is2")