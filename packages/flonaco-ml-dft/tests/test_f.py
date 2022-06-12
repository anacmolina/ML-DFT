import numpy as np

import gpaw.mpi as mpi
import torch
import pickle
import pandas as pd

from flonacomldft.dft_utils import (
    Angles_transformation,
    Structure
)

import torch.backends.cudnn as cudnn

from flonacomldft.sampling import run_metropolis

from flonacomldft.md_utils import (
    get_path,
    load_from_pickle,
    load_is_csv, 
    shuffle_arr,
    get_is1,
    get_is2,
    get_internal_coordinates,
    run_molecular_dynamics
)

from flonacomldft.mixture import Mixture

from flonacomldft.train_from_data import train
from flonacomldft.training_utils import init_model, run_NF

rank = mpi.world.rank

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
xi_is1, ui_is1, ci_is1, _is1 = run_NF(get_is1(), 2, 'is1', device, None, 0)

mpi.world.barrier()
print("RESULTS ", rank)

print(rank, xi_is1, xi_is1.shape)
