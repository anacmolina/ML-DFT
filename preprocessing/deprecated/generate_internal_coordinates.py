# libraries
import argparse

import torch
from ase.io.trajectory import Trajectory
from flonacomldft.internal_coordinates import (
    add_phase,
    get_construction_table,
    Coordinates_mapping,
    save_internal_coordinates_to_csv
)
from flonacomldft.utils.io_utils import get_path

#TODO: add device and dtype

# define arguments argparser
parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('-file', '--file', type=str,)
parser.add_argument('-isomer', '--isomer-label', type=int,)
parser.add_argument('-etype', '--energy-type', type=str,)
parser.add_argument('-T', '--temperature', type=int, default=350)
parser.add_argument('-N', '--num-samples', type=int, default=None)
parser.add_argument('-low', '--low-index', type=int, default=0)

args = parser.parse_args()

dim = 12
isomer_label = args.isomer_label
N = args.num_samples

# load trajectory
input_file = args.file
traj = Trajectory(input_file)[args.low_index:]

# compute internal coordinates
coord_mapping = Coordinates_mapping(etype=args.energy_type)
zmats = coord_mapping.get_internal_from_trajectory(traj, 
                                                    isomer=isomer_label, 
                                                    temperature=args.temperature, 
                                                    max_samples=N).detach()

if isomer_label == 0:
    zmats[:, 11][zmats[:, 11]>0] = zmats[:, 11][zmats[:, 11]>0].apply_(add_phase)

# save internal coordinates to csv
output_file_zmat = args.file.split('/')[-1].split('.')[0]+ '_zmats' + '.csv'
folder = args.file.split('/')[-1].split('.')[0].split('_')[2] + '/internal_coordinates'
save_internal_coordinates_to_csv(zmats, 
                                construction_table=get_construction_table(), 
                                filename=output_file_zmat, 
                                path=get_path() + '/' + folder)

# compute real centered coordinates
coord_xs, logdetjac_xs, energies_xs = coord_mapping.get_real_centered_from_internal(zmats[:, :dim], 
                                                    isomer=isomer_label, 
                                                    temperature=350,
                                                    energies=zmats[:, dim],
                                                    logdetjacs=zmats[:, dim+2])

xs = torch.cat([coord_xs, 
                energies_xs.reshape(-1, 1),
                torch.ones(coord_xs.shape[0]).reshape(-1, 1)*isomer_label,
                logdetjac_xs.reshape(-1, 1),
                zmats[:, -2:]], dim=1)

# save real centered coordinates to csv
output_file_rc = args.file.split('/')[-1].split('.')[0]+ '_xs' + '.csv'
columns = ['rc{:d}' for i in range(dim)]
save_internal_coordinates_to_csv( xs, 
                                columns=columns, 
                                filename=output_file_rc, 
                                path=get_path() + '/' + folder)





