#import warnings
#warnings.filterwarnings("ignore")

import os
import argparse

from ase.io import read

import torch
import pandas as pd
from flonacomldft.internal_coordinates import Coordinates_mapping
from flonacomldft.internal_coordinates import save_internal_coordinates_to_csv
from flonacomldft.utils.io_utils import save_ase_molecules_as_traj 

# define arguments argparser
parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('-path', '--path', type=str,)
parser.add_argument('-isomer', '--isomer-label', type=int,)
parser.add_argument('-etype', '--energy-type', type=str,)
parser.add_argument('-T', '--temperature', type=int, default=350)
parser.add_argument('-N', '--num-samples', type=int, default=None)
parser.add_argument('-low', '--low-index', type=int, default=None)

args = parser.parse_args()

coord_mapping = Coordinates_mapping(etype=args.energy_type)

comps_id = int(args.path.split('/')[0].split('_')[-1])

file_names = os.listdir(args.path)
file_names.remove('init_calc.out')

if args.low_index is None:
    low = 0
else:
    low = args.low_index

if args.num_samples is None:
    N = len(file_names)
else:
    N = args.num_samples

molecules = [read(args.path + '/' + file_name) for file_name in file_names[low:low+N]]
zmats = coord_mapping.get_internal_from_trajectory(molecules, isomer=args.isomer_label, temperature=args.temperature, max_samples=N).detach()

print(zmats.shape)
 
save_internal_coordinates_to_csv(zmats, 
                                construction_table=coord_mapping.construction_table, 
                                filename='is{:d}_samples_zmat_md_{:d}.csv'.format(args.isomer_label, comps_id), 
                                add_cvs=True,
                                path=args.path)

save_ase_molecules_as_traj(molecules, 'is{:d}_samples_md_{:d}.traj'.format(args.isomer_label, comps_id), path=args.path)

xs_rc = coord_mapping.get_real_centered_from_internal(zmats[:, :12], 
                                                      isomer=args.isomer_label,
                                                      temperature=args.temperature, 
                                                      energies=zmats[:, 12])

cols = ['rc_{:d}'.format(i) for i in range(12)] + ['potential_energy'] + ['isomer']

xs_rc = torch.cat((xs_rc[0], xs_rc[2].reshape(-1, 1), torch.ones((N, 1))*args.isomer_label), dim=1)

df = pd.DataFrame(xs_rc.detach().numpy(), columns=cols)

df.to_csv(args.path + '/is{:d}_samples_rc_md_{:d}.csv'.format(args.isomer_label, comps_id), index=False)
