### Import modules
import argparse

from ase.io.trajectory import Trajectory
from flonacomldft.internal_coordinates import (
    add_phase,
    get_construction_table,
    Coordinates_mapping,
    save_internal_coordinates_to_csv
)
from flonacomldft.utils.io_utils import get_path

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('-file', '--file', type=str)
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-N', '--num-samples', type=int, default=None)

args = parser.parse_args()

mode_label = args.mode_label
N = args.num_samples

### Load trajectory
input_file = args.file
traj = Trajectory(input_file)[7000:]

### Generate internal coordinates
coord_mapping = Coordinates_mapping()
zmats = coord_mapping.get_internal_from_trajectory(traj, isomer=mode_label, temperature=300, max_samples=N)
zmats = zmats.detach()

if mode_label == 0:
    zmats[:, 11][zmats[:, 11]>0] = zmats[:, 11][zmats[:, 11]>0].apply_(add_phase)

### Save internal coordinates
output_file = args.file.split('/')[-1].split('.')[0] + '.csv'
save_internal_coordinates_to_csv(zmats, get_construction_table(), filename=output_file)
