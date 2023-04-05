#TODO: add docstring

import time
import argparse
import gpaw.mpi as mpi

from flonacomldft.utils.io_utils import (
    load_pickle_file,
    save_ase_molecules_as_traj,
    get_project_path,
    save_json_args
)

from flonacomldft.dft_calculator import DFTCalculator
from flonacomldft.parallel import set_seed

from flonacomldft.internal_coordinates import (
    Coordinates_mapping,
    get_construction_table,
    save_internal_coordinates_to_csv
)

# set equal seed for all ranks for parallel computations

num_seed = set_seed()

# for naming files
date_start = time.strftime('%H:%M:%S %d-%m-%Y')

# Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-N', '--num-samples', type=int, default=1000)
parser.add_argument('-id', '--id', type=int, default=0)
parser.add_argument('-inpath', '--input-path', type=str, default='database/')
parser.add_argument('-outpath', '--output-path', type=str, default='database/')

args = parser.parse_args()
args.date_start = date_start
args.num_seed = num_seed

mode_label = args.mode_label
N = args.num_samples
input_path = get_project_path() + args.input_path
output_path = get_project_path() + args.output_path
filename = 'is{:d}_flow_dic_training_{:d}.pkl'.format(mode_label, args.id)

flow_model = load_pickle_file(filename, path=input_path)['model']

mpi.world.barrier()
coord_mapping = Coordinates_mapping()
xs_samples = flow_model.sample(N)
zs_samples, logdetjacs = coord_mapping.get_internal_from_real_centered(xs_samples, isomer=mode_label)

mpi.world.barrier()
calculator = DFTCalculator()
calculator.initialize_calculator(foldername='DFTComputations_is{:d}_{:d}'.format(mode_label, args.id))

flow_configurations = []

for i, zmat_sample in enumerate(zs_samples):
    molecule = coord_mapping.build_molecule_from_zmat(zmat_sample.detach())
    calculator.calculate_potential_energy(molecule, 
                            filename='ag6_flow_is{:d}_{:d}.out'.format(mode_label, i))

    flow_configurations.append(molecule)

mpi.world.barrier()
internal_coordinates = coord_mapping.get_internal_from_trajectory(flow_configurations, isomer=mode_label, temperature=300)

mpi.world.barrier()
save_ase_molecules_as_traj(flow_configurations, 'ag6_flow_is{:d}_{:d}.traj'.format(mode_label, args.id), output_path)

mpi.world.barrier()
save_internal_coordinates_to_csv(internal_coordinates,
            get_construction_table(),
            filename='ag6_flow_zmat_is{:d}_{:d}.csv'.format(mode_label, args.id), path=output_path)

mpi.world.barrier()
date_end = time.strftime('%H:%M:%S %d-%m-%Y')
args.date_end = date_end
args.algorithm = 'samples_from_flow.py'

print(args)

mpi.world.barrier()
save_json_args(args)