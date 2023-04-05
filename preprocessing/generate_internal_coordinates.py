#TODO: Add docstring

import argparse

from ase.io.trajectory import Trajectory
from flonacomldft.utils.io_utils import get_project_path
from flonacomldft.internal_coordinates import (
    add_phase,
    get_construction_table,
    Coordinates_mapping,
    save_internal_coordinates_to_csv
)

# Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-N', '--num-samples', type=int, default=None)
parser.add_argument('-path', '--folder-path', type=str, default='database/')
parser.add_argument('-id', '--id', type=int, default=0)

args = parser.parse_args()

mode_label = args.mode_label
N = args.num_samples

input_file = get_project_path() + args.folder_path + 'ag6_md_is{:d}_{:d}.traj'.format(mode_label, args.id)

traj = Trajectory(input_file) 

coord_mapping = Coordinates_mapping()
zmats = coord_mapping.get_internal_from_trajectory(traj, isomer=mode_label, temperature=300, max_samples=N)
zmats = zmats.detach()

if mode_label == 0:
    zmats[:, 11][zmats[:, 11]>0] = zmats[:, 11][zmats[:, 11]>0].apply_(add_phase)

output_file = args.folder_path + 'ag6_md_zmat_is{:d}_{:d}.csv'.format(mode_label, args.id)
save_internal_coordinates_to_csv(zmats, get_construction_table(), filename=output_file)
