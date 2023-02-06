import copy
import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from ase.visualize import view

from flonacomldft.internal_coordinates import Coordinates_mapping

from flonacomldft.utils.io_utils import load_csv_file

mode_label = 0

data = load_csv_file("datasets/is{:d}_flow_train.csv".format(mode_label))
coord_mapping = Coordinates_mapping()

n = 1

# int_coords
zs = data.clone()[:, :12][:n]
logdetjac = data.clone()[:, 14][:n] 
energies = data.clone()[:, 12][:n]

ag6 = coord_mapping.build_molecule_from_zmat(zs[0])

print('zmat\n')
print(zs, logdetjac, energies)

# centering around minima and real space (tan)
zs_reals, logdetjac_reals, energies_reals = coord_mapping.get_real_centered_from_internal(zs, logdetjac, isomer=0, energies=energies)

print('reals\n')
print(zs_reals, logdetjac_reals, energies_reals)

# back to internal coords
xs_back, logdetjac_back, energies_back = coord_mapping.get_internal_from_real_centered(zs_reals, logdetjac_reals, isomer=0, energies=energies_reals)

print('zmat\n')
print(xs_back, logdetjac_back, energies_back)

# from real_centered to cartesian

ag6_back = coord_mapping.build_molecule_from_zmat(xs_back[0])

view(ag6_back)
