import os
import argparse

from ase.io import read

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

file_names = os.listdir(args.path)

print(args.path)
print(len(file_names))

if args.low_index is None:
    low = 0
else:
    low = args.low_index

if args.num_samples is None:
    N = len(file_names)
else:
    N = args.num_samples



molecules = [read(args.path + '/' + file_name) for file_name in file_names[low:low+N]]
print(len(molecules))
zmats = coord_mapping.get_internal_from_trajectory(molecules, isomer=args.isomer_label, temperature=args.temperature, max_samples=N).detach()

print(zmats.shape)
 
save_internal_coordinates_to_csv(zmats, 
                                construction_table=coord_mapping.construction_table, 
                                filename='is{:d}_{:s}_mlp.csv'.format(args.isomer_label, args.energy_type), 
                                add_cvs=True)

save_ase_molecules_as_traj(molecules, 'is{:d}_{:s}_mlp.traj'.format(args.isomer_label, args.energy_type))
