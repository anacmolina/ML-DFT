import time
import copy

import torch
import numpy as np
import matplotlib.pyplot as plt

from flonacomldft.dft_utils import (
    Angles_mapping,
    shuffle_arr
)

from flonacomldft.data_utils import (
    get_path,
    load_zmat_csv,
    save_pickle_file
)

from flonacomldft.md_mcmc import (
    get_mix_data
)

from flonacomldft.real_nvp_mlp import RealNVP_MLP
from flonacomldft.train_from_data import train
from flonacomldft.mixture import Mixture

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype=torch.float32

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)

# Loading data

data_is1 = load_zmat_csv('is1')
data_is2 = load_zmat_csv('is2')

# Shuffling data
#ind = torch.randperm(data_is1.shape[0]+data_is2.shape[0])
#data = shuffle_arr((data_is1, data_is2), ind)

# Mapping

def train_isomer(data):
    M = Angles_mapping()
    M.inv_mapping(data[:, :-1])

    cov = torch.cov(data.T)
    mean = data.mean(0)

    print("Normalizing flow training")

    args_rnvp = {
        'dim': data.shape[1],
        'n_realnvp_block': 15,
        'block_depth': 1,
        # 'args_prior': {'type': 'standn'}, # standard Gaussian base
        'args_prior': {'type': 'white', 'cov': cov, 'mean': mean}, # Gaussian with non-trival mean and covariance for base
        'init_weight_scale': 1e-6,
    }

    model = RealNVP_MLP(args_rnvp['dim'], 
                    args_rnvp['n_realnvp_block'],
                    args_rnvp['block_depth'],
                    init_weight_scale=args_rnvp['init_weight_scale'],
                    prior_arg=args_rnvp['args_prior'],
                    device=device)

    model_init = copy.deepcopy(model)

    _ = train(model, 
            data,
            n_iter=2500,
            lr=1e-3,
            bs=10,
            use_scheduler=False,
            step_schedule=100,
            args_loss={'type': 'fwd', 'samp': 'direct'},
            estimate_tau=False,
            return_all_xs=True,
            save_splits=10,
            grad_clip=1e4)

    print("Done\n")
    return _

is1_training = train_isomer(data_is1)
is2_training = train_isomer(data_is2)

save_pickle_file(is1_training, "nf_energy_training_is1")
save_pickle_file(is2_training, "nf_energy_training_is2")

model_is1 = is1_training['models'][-1]
model_is2 = is2_training['models'][-1]

mixture = Mixture([model_is1, model_is2], torch.tensor([0.75, 0.25]).detach())

save_pickle_file(mixture, 'nf_energy_mixture')