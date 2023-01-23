import copy
import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from flonacomldft.internal_coordinates import Coordinates_mapping

from flonacomldft.utils.io_utils import load_csv_file

data = load_csv_file("datasets/is1_md_train.csv")
coord_mapping = Coordinates_mapping()

n = 1

# int coords
zs = data.clone()[:, :12][:n]
logdetjac = data.clone()[:, 12][:n] 
energies = data.clone()[:, 13][:n]

print('\n')
print(zs, logdetjac, energies)

# centering around and reals
zs_reals, logdetjac_reals, energies_reals = coord_mapping.get_real_centered_from_internal(zs, logdetjac, isomer=0, energies=energies)

print('\n')
print(zs_reals, logdetjac_reals, energies_reals)

xs_back, logdetjac_back, energies_back = coord_mapping.get_internal_from_real_centered(zs_reals, logdetjac_reals, isomer=0, energies=energies_reals)

print('\n')
print(xs_back, logdetjac_back, energies_back)

