import time
import copy

import torch
import numpy as np 
import pandas as pd 

from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.internal_coordinates import Angles_mapping

from flonacomldft.utils.data_utils import get_path, save_pickle_file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

date = time.strftime("%d-%m-%Y")
random_id = np.random.randint(100)
torch.manual_seed(random_id)

print("Date: {}".format(date))
print("Random_id: {}".format(str(random_id)))

df = pd.read_csv(get_path() + "is1_lcao_zmat.csv")
U = df.energies
X = df.drop(["energies"], axis=1)

n = 3000 #-1

x_tensor = torch.from_numpy(X[:n].to_numpy()).float()
U_tensor = torch.from_numpy(U[:n].to_numpy()).float()

print("Labels: {}".format(x_tensor.shape[1]))
print("Samples: {}".format(x_tensor.shape[0]))

Angles_mapping().inv_mapping(x_tensor)

cov = torch.cov(x_tensor.T)
mean = x_tensor.mean(0)

args_rnvp = {
    "dim": x_tensor.shape[1],
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

flow_hyperparams = {'n_iter': 500,
    'lr': 5e-4,
    'use_scheduler': False,
    'step_schedule': 100,
    'save_splits': 10,
    'grad_clip': 1e4}

_ = train_flow(
    model,
    x_tensor,
    **flow_hyperparams,
)

save_pickle_file(_, "flow_init_is1")