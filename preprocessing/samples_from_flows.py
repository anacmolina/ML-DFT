#TODO: add docstring

### Import modules
import time
import argparse
import gpaw.mpi as mpi

from flonacomldft.utils.io_utils import (
    load_pickle_file,
    save_ase_molecules_as_traj,
    get_process_id,
    save_json_args
)

from flonacomldft.dft_calculator import DFTCalculator
from flonacomldft.parallel import set_seed

from flonacomldft.internal_coordinates import (
    Coordinates_mapping,
    get_construction_table,
    save_internal_coordinates_to_csv
)

### Set equal seed for all ranks for parallel computations
num_seed = set_seed()

### For naming files
date_start = time.strftime('%Y-%m-%d %H:%M:%S')
process_id = get_process_id(date_start)

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-N', '--num-samples', type=int, default=1000)
parser.add_argument('-id', '--id', type=int, default=0)
parser.add_argument('-path', '--file-path', type=str)

args = parser.parse_args()
args.date_start = date_start
args.num_seed = num_seed

### Load flow model
mode_label = args.mode_label
N = args.num_samples

flow_model = load_pickle_file(args.file_path)['model']

mpi.world.barrier()

### Sample new configurations from flow model
xs_samples = flow_model.sample(N)

### Get real center coordinates
coord_mapping = Coordinates_mapping()
zs_samples, logdetjacs = coord_mapping.get_internal_from_real_centered(xs_samples, isomer=mode_label)

mpi.world.barrier()

### Compute energies with DFT
calculator = DFTCalculator()
calculator.initialize_calculator(foldername='DFTComputations_is{:d}_{:d}'.format(mode_label, args.process_id))

flow_configurations = []

for i, zmat_sample in enumerate(zs_samples):
    molecule = coord_mapping.build_molecule_from_zmat(zmat_sample.detach())
    calculator.calculate_potential_energy(molecule, 
                            filename='is{:d}_samples_{:d}.out'.format(mode_label, i))

    flow_configurations.append(molecule)

mpi.world.barrier()

### Get internal coordinates of sampled configurations
internal_coordinates = coord_mapping.get_internal_from_trajectory(flow_configurations, isomer=mode_label, temperature=300)

mpi.world.barrier()

### Save data as traj file
save_ase_molecules_as_traj(flow_configurations, 'is{:d}_samples_{:d}.traj'.format(mode_label, args.process_id))

mpi.world.barrier()

### Save internal coordinates as csv file
save_internal_coordinates_to_csv(internal_coordinates,
            get_construction_table(),
            filename='is{:d}_samples_{:d}.csv'.format(mode_label, args.process_id))

mpi.world.barrier()

date_end = time.strftime('%Y-%m-%d %H:%M:%S')
args.date_end = date_end
args.algorithm = 'samples_from_flow.py'

mpi.world.barrier()

### Save arguments
save_json_args(args)