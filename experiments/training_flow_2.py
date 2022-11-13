from ase.parallel import parprint as print

import time
import copy
import json

import torch
import numpy as np
import pandas as pd

from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.internal_coordinates import Angles_mapping

from flonacomldft.utils.data_utils import get_path, save_pickle_file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

f = open('flow_specs.out', 'a')

date = time.strftime("%d-%m-%Y")
random_id = np.random.randint(100)

f.write("FLONACO ML-DFT\n\n")
f.write("Flow training\n\n")
f.write("Date: {}\n".format(date))
f.write("Random_seed: {}\n\n".format(str(random_id)))

f.write("Ag6 isomer: planar\n")

torch.manual_seed(random_id)

df = pd.read_csv(get_path() + "is1_lcao_zmat.csv")
U = df.energies
X = df.drop(["energies"], axis=1)

n = -1

x_tensor = torch.from_numpy(X[:n].to_numpy()).float()
U_tensor = torch.from_numpy(U[:n].to_numpy()).float()

f.write("Samples: {}, Labels: {}\n\n".format(x_tensor.shape[0], x_tensor.shape[1]))

Angles_mapping().inv_mapping(x_tensor)

cov = torch.cov(x_tensor.T)
mean = x_tensor.mean(0)

args_rnvp = {
    "dim": x_tensor.shape[1],
    "n_realnvp_block": 5,
    "block_depth": 1,
    # 'args_prior': {'type': 'standn'}, # standard Gaussian base
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

f.write("flow NN architecture: \n")
f.write("\t n_realnvp_block: {}\n".format(args_rnvp['n_realnvp_block']))
f.write("\t block_depth: {}\n".format(args_rnvp['block_depth']))

f.write("\n")

n_iter_ = 1000
lr_ = 5e-3

f.write("flow training hyperparameters: \n")
f.write("\t n_iter: {}\n".format(n_iter_))
f.write("\t lr: {}\n".format(lr_))

f.write("\n")

model_init = copy.deepcopy(model)

#TODO: writing training values in the output file
#TODO: add way to add isomer

_ = train_flow(
    model,
    x_tensor,
    n_iter=n_iter_,
    lr=lr_,
    bs=100,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
)

f.write("DONE!")

f.close()
save_pickle_file(_, "flow_trained")
