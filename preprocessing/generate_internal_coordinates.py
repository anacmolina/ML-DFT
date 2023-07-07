# libraries
import argparse

from ase.io.trajectory import Trajectory
from flonacomldft.internal_coordinates import (
    add_phase,
    get_construction_table,
    Coordinates_mapping,
    save_internal_coordinates_to_csv
)
from flonacomldft.utils.io_utils import get_path

# define arguments argparser
parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('-file', '--file', type=str)
parser.add_argument('-isomer', '--isomer-label', type=int, default=0)
parser.add_argument('-N', '--num-samples', type=int, default=None)
parser.add_argument('-low', '--low-index', type=int, default=0)

args = parser.parse_args()

isomer_label = args.isomer_label
N = args.num_samples

# load trajectory
input_file = args.file
traj = Trajectory(input_file)[args.low_index:]

# compute internal coordinates
coord_mapping = Coordinates_mapping()
zmats = coord_mapping.get_internal_from_trajectory(traj, isomer=isomer_label, temperature=300, max_samples=N).detach()

if isomer_label == 0:
    zmats[:, 11][zmats[:, 11]>0] = zmats[:, 11][zmats[:, 11]>0].apply_(add_phase)

# save internal coordinates to csv
output_file = args.file.split('/')[-1].split('.')[0] + '.csv'
save_internal_coordinates_to_csv(zmats, get_construction_table(), filename=output_file)

#TODO: add device and dtype
#TODO: add collective variables to the csv file