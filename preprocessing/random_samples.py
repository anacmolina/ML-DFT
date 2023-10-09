import warnings
warnings.filterwarnings("ignore")

import time
import argparse
import numpy as np
import torch

from flonacomldft.utils.data_processing import load_datasets
from flonacomldft.utils.io_utils import (
    set_str_date_to_int,
    save_ase_molecules_as_traj
    )

from flonacomldft.internal_coordinates import (
    Coordinates_mapping, 
    save_internal_coordinates_to_csv,
    get_construction_table
    )

from flonacomldft.models.real_nvp import RealNVP_MLP

# define arpase arguments
parser = argparse.ArgumentParser(description='Prepare experiment')
# execution params
parser.add_argument('-threads', '--threads', type=int, default=None)
parser.add_argument('-pid', '--process-id', type=int, default=None)
parser.add_argument('-rs', '--random-seed', type=int, default=None)
parser.add_argument('-path', '--folder-path', type=str,)
# simulation params
parser.add_argument('-isomer', '--isomer-label', type=int,)
parser.add_argument('-etype', '--energy-type', type=str,)
parser.add_argument('-N', '--num-samples', type=int,)
parser.add_argument('-T', '--temperature', type=float, default=350)
# flow params
parser.add_argument('-nb', '--n-blocks', type=int, default=4)
parser.add_argument('-nodes', '--hidden-dim', type=int, default=64)
parser.add_argument('-layers', '--hidden-depth', type=int, default=3)

args = parser.parse_args()


if args.energy_type == 'emt':

    num_seed = np.random.randint(0, 100)
    date_start = set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))

    args.random_seed = num_seed
    args.date_start = str(date_start)

    if args.process_id is None:
        args.process_id = date_start

    from flonacomldft.dft_calculator import EMTCalculator

    calculator = EMTCalculator()

elif args.energy_type == 'dft':
    
    # parallelization set up
    import gpaw.mpi as mpi

    ranks = np.arange(0, mpi.world.size)
    rank = mpi.rank
    comm = mpi.world.new_communicator(ranks)

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

    args.random_seed = num_seed
    args.date_start = str(date_start)

    from flonacomldft.dft_calculator import DFTCalculator

    calculator = DFTCalculator()
    calculator.initialize_calculator()

else:
    raise ValueError('Energy type not supported')

# torch settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.threads is not None:
    torch.set_num_threads(args.threads)

# set random seed
torch.manual_seed(int(args.random_seed))

print('device: ', device)
print('threads: ', torch.get_num_threads())
print('seed: ', args.random_seed, num_seed)
print('date_start: ', args.date_start)


xs = load_datasets(args.folder_path, 
                   args.isomer_label, 
                   name='flow', 
                   real_centered=True, 
                   temperature=args.temperature)
xs =  torch.cat([xs['train'], xs['test']])

cov = torch.cov(xs[:, :12].T).detach() + 1e-5 * torch.eye(xs[:, :12].shape[1]).detach()
model = RealNVP_MLP(dim=xs[:, :12].shape[1],
                    n_blocks=args.n_blocks,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    base_cov=cov,
                    hidden_dim=args.hidden_dim,
                    hidden_depth=args.hidden_depth,
                    device=device,
                    )

new_xs = model.sample(args.num_samples)
us_xs = torch.cat([torch.zeros(args.num_samples, 1), torch.ones(args.num_samples, 1)*args.isomer_label], dim=1).detach()

coord_mapping = Coordinates_mapping(etype=args.energy_type)

molecules = []

for i, x in enumerate(new_xs):
    
    molecule, logdetjac = coord_mapping.build_molecule_from_real_centered(x[:12].reshape(1, -1).detach(), isomer=args.isomer_label, temperature=args.temperature)
    
    calculator_input = {'atoms': molecule,}
    if args.energy_type == 'dft':
        calculator_input['filename'] = 'is{:d}_samples_{:d}.out'.format(args.isomer_label, i)
    
    u = calculator.calculate_potential_energy(**calculator_input)

    molecules.append(molecule)

    us_xs[i, 0] = coord_mapping.compute_energy_in_new_frame(u, logdetjac*(-1), temperature=args.temperature)

new_xs = torch.cat([new_xs, us_xs], dim=1)

new_zmat, logdetjac_zmat, us_zmat = coord_mapping.get_internal_from_real_centered(new_xs[:, :12].detach(), 
                                                         isomer=args.isomer_label,
                                                         energies=new_xs[:, 12].detach(),)

CV = coord_mapping.get_collective_variables_from_trajectory(molecules)

new_zmat = torch.cat([new_zmat, 
                      us_zmat.reshape(-1, 1),
                      new_xs[:, 13].reshape(-1, 1), 
                      logdetjac_zmat.reshape(-1, 1),
                      CV], dim=1)

save_ase_molecules_as_traj(molecules, 'is{:d}_emt_mlp.traj'.format(args.isomer_label))

save_internal_coordinates_to_csv(new_zmat,
            get_construction_table(),
            add_cvs=True,
            filename='is{:d}_emt_mlp.csv'.format(args.isomer_label))
