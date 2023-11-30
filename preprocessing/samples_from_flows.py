#TODO: Review file

### Import modules
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch

from flonacomldft.utils.io_utils import (
    load_pickle_file,
    save_ase_molecules_as_traj,
    save_json_args,
    set_str_date_to_int,
    get_project_path
)

from flonacomldft.dft_calculator import DFTCalculator


from flonacomldft.internal_coordinates import (
    Coordinates_mapping,
    get_construction_table,
    save_internal_coordinates_to_csv,
)

# parallelization set up
import gpaw.mpi as mpi

ranks = np.arange(0, mpi.world.size)
rank = mpi.rank
comm = mpi.world.new_communicator(ranks)

print('ranks: ', mpi.world.size)

num_seed = np.array([0])
date_start = np.array([0])

# only rank 0 generates the seed and date_start
if rank == 0:
    num_seed = np.random.randint(0, 100, (1,))
    date_start = np.array([set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))])

mpi.world.barrier()

comm.broadcast(num_seed, 0)
comm.broadcast(date_start, 0)

num_seed = num_seed[0]
date_start = date_start[0]

print('seed: ', num_seed, rank)
print('date_start: ', date_start, rank)


### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-isomer', '--isomer-label', type=int, default=0)
parser.add_argument('-ns', '--num-samples', type=int, default=5)
parser.add_argument('-nstage', '--n-stage', type=int, default=-2)
parser.add_argument('-pid', '--process-id', type=int, default=date_start)
parser.add_argument('-path', '--file-path', type=str, default=get_project_path() + '/3-flows-md')
parser.add_argument('-flowsID', '--flows-id', type=int)
parser.add_argument('-etype', '--energy-type', type=str)
parser.add_argument('-T', '--temperature', type=float, default=350)

args = parser.parse_args()
args.date_start = date_start
args.num_seed = num_seed

folder_path = args.file_path + '/results_flow_is{:d}_{:d}'.format(args.isomer_label, args.flows_id)
file = 'is{:d}_flow_dic_{:d}.pkl'.format(args.isomer_label, args.flows_id)
#torch.set_num_threads(len(ranks))

torch.manual_seed(num_seed)


### Load flow model
isomer_label = args.isomer_label
N = args.num_samples

flow_model = load_pickle_file(file, path=folder_path)['model']

mpi.world.barrier()

#folder_to_save_results = 'results_fe_{:s}_isomer_{:d}_{:d}'.format(args.energy_type, isomer_label, args.process_id)
#path_to_save_results = os.getcwd() + '/' + folder_to_save_results

#if rank == 0:
#    
#    if not os.path.exists(path_to_save_results):
#        
#        os.makedirs(path_to_save_results)
#        print('folder created: ', path_to_save_results, rank)

### Sample new configurations from flow model
xs_samples = flow_model.sample(N)

### Get real center coordinates
coord_mapping = Coordinates_mapping(etype=args.energy_type)
zs_samples, logdetjacs = coord_mapping.get_internal_from_real_centered(xs_samples, isomer=isomer_label)

mpi.world.barrier()

### Compute energies with DFT
calculator = DFTCalculator()
calculator.initialize_calculator(foldername=folder_path + '/DFTComputations_is{:d}_{:d}'.format(isomer_label, args.process_id))

flow_configurations = []

for i, zmat_sample in enumerate(zs_samples):
    molecule = coord_mapping.build_molecule_from_zmat(zmat_sample.detach())
    calculator.calculate_potential_energy(molecule, 
                            filename='is{:d}_samples_{:d}.out'.format(isomer_label, i))

    flow_configurations.append(molecule)

mpi.world.barrier()

### Get internal coordinates of sampled configurations
internal_coordinates = coord_mapping.get_internal_from_trajectory(flow_configurations, isomer=isomer_label, temperature=args.temperature)

mpi.world.barrier()

### Save data as traj file
save_ase_molecules_as_traj(flow_configurations, 'is{:d}_samples_{:d}.traj'.format(isomer_label, args.flows_id), path=folder_path)

mpi.world.barrier()

### Save internal coordinates as csv file
save_internal_coordinates_to_csv(internal_coordinates,
            construction_table=get_construction_table(),
            filename='is{:d}_samples_zmat_{:d}.csv'.format(isomer_label, args.process_id),
            path=folder_path)

mpi.world.barrier()

date_end = time.strftime('%Y-%m-%d %H:%M:%S')
args.date_end = date_end
args.algorithm = 'samples_from_flow.py'

mpi.world.barrier()

### Save arguments
save_json_args(args, 'samples_from_flow', args.process_id, path=folder_path)

mpi.world.barrier()

xs_rc = coord_mapping.get_real_centered_from_internal(internal_coordinates[:, :12], 
                                                      isomer=isomer_label,
                                                      temperature=args.temperature, 
                                                      energies=internal_coordinates[:, 12])

mpi.world.barrier()

cols = ['rc_{:d}'.format(i) for i in range(12)] + ['potential_energy'] + ['isomer']

xs_rc = torch.cat((xs_rc[0], xs_rc[2].reshape(-1, 1), torch.ones((N, 1))*isomer_label), dim=1)

df = pd.DataFrame(xs_rc.detach().numpy(), columns=cols)

df.to_csv(folder_path + '/is{:d}_samples_rc_{:d}.csv'.format(isomer_label, args.process_id), index=False)

import matplotlib.pyplot as plt
from flonacomldft.utils.plots import plot_energy_surface

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
plot_energy_surface(fig=fig, ax=ax)
ax.scatter(internal_coordinates[:, -2].detach().numpy(), 
           internal_coordinates[:, -1].detach().numpy(), 
           c='orange')
plt.savefig(folder_path + '/is{:d}_samples_cvs_{:d}.png'.format(isomer_label, args.process_id))

