# TODO: add description
# TODO: add logging file

# load libraries
from os import getcwd
import argparse
import time

# load scientific libraries
import torch
import numpy as np
import matplotlib.pyplot as plt

# load DFT libraries 
from ase.units import (kB, fs)
from gpaw import GPAW
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.io.trajectory import Trajectory

# load flonaco utils
from flonacomldft.utils.silver_isomers_utils import (
    get_molecule_isomer_minima,
    get_calculator_params,
)
from flonacomldft.utils.io_utils import save_json_args
from flonacomldft.dft_calculator import run_molecular_dynamics
from flonacomldft.utils.io_utils import (
    set_str_date_to_int,
    set_int_date_to_str,
)
from flonacomldft.utils.plots import set_plot_sequential_data

# load parallel libraries
#from ase.parallel import parprint as print
import gpaw.mpi as mpi

# set up MPI
ranks = np.arange(0, mpi.world.size)
rank = mpi.world.rank
comm = mpi.world.new_communicator(ranks)

# get initial date and set random seed
num_seed = np.array([0])
date_start = np.array([0])

# only rank 0 generates the seed and date_start

if rank == 0:
    num_seed = np.random.randint(0, 100, (1,))
    date_start = np.array([set_str_date_to_int(time.strftime("%Y-%m-%d %H:%M:%S"))])

mpi.world.barrier()

# broadcast seed and date_start to all ranks
comm.broadcast(num_seed, 0)
comm.broadcast(date_start, 0)

# final seed and date_start
num_seed = num_seed[0]
date_start = set_int_date_to_str(date_start[0])

# set up parser 

parser = argparse.ArgumentParser(description='Run molecular dynamics simulation')
# system parameters
parser.add_argument('-symbs', '--symbols', type=str, default='Ag', help='Symbols of the molecule')
parser.add_argument('-isomer', '--isomer-label' , type=str, help='Isomer label')
parser.add_argument('-cell', '--cell', type=float, help='Cell size')
parser.add_argument('-vacuum', '--vacuum', type=float, help='Vacuum size')
parser.add_argument('-pbc', '--pbc', type=bool, default=True, help='Periodic boundary conditions')
parser.add_argument('-gpwmd', '--gpaw-mode', type=str, default='LCAO', help='GPAW mode')
parser.add_argument('-etype', '--energy-type', type=str, default='dft', help='Energy type')
#TODO: vefify if the energy type EMT is working
#TODO: change energy type format to potential type
# molecular dynamics parameters
parser.add_argument('-tt', '--thermostat-type', type=str, help='Thermostat type')
parser.add_argument('-ns', '--n-steps', type=int, default=5)
parser.add_argument('-ts', '--time-step', type=float, default=5)
parser.add_argument('-ninterval', '--n-interval', type=int, default=1)
parser.add_argument('-T', '--temperature', type=float, default=300)
parser.add_argument('-taut', '--taut', type=float, default=None) #50
parser.add_argument('-f', '--friction', type=float, default=None) #2e-3
parser.add_argument('-ap', '--andersen-prob', type=float, default=None) #2e-3
parser.add_argument('-pid', '--process-id', type=str, default=date_start)
parser.add_argument('-s', '--seed', type=int, default=num_seed)

args = parser.parse_args()
args.date_start = date_start

# get molecule
molecule = get_molecule_isomer_minima('{:s}'.format(args.symbols),
                                      '{:s}'.format(args.isomer_label))

# set cell or vacuum
if args.cell is not None and args.vacuum is not None:
    
    raise ValueError("Cell and vacuum cannot be both set, just one must be set")

elif args.cell is not None and args.vacuum is None:
    
    molecule.set_cell([args.cell, args.cell, args.cell])
    molecule.set_pbc(args.pbc)
    molecule.center()

elif args.cell is None and args.vacuum is not None:
    
    molecule.set_pbc(args.pbc)
    molecule.center(vacuum=args.vacuum)

else:

    raise ValueError("Cell and vacuum cannot be both None, just one must be set")

filename = "{:s}_{:s} {:s} {:s}".format(args.symbols, 
                                        args.isomer_label, 
                                        args.thermostat_type, 
                                        args.process_id) 

# set calculator
if args.energy_type == 'dft':
    
    

    params_calc = get_calculator_params(name=args.gpaw_mode)
    params_calc['txt'] = filename + ".out"

    mpi.world.barrier()

    print(params_calc)
    calc = GPAW(**params_calc)

#TODO: Review functionality
#if args.energy_type == 'emt':
#    
#    from ase.calculators.emt import EMT
#    
#    MOLECULE = 'emt_is{:d}'.format(args.isomer_label)
#    calc = EMT()

# TODO: add logging file
MD_PARAMS = {'thermostat': args.thermostat_type,
             'timestep': args.time_step * fs,
             'temperature_K': args.temperature,
             'taut': args.taut,
             'andersen_prob': args.andersen_prob,
             'friction': args.friction,
}

if MD_PARAMS['thermostat'] == 'berendsen':
    MD_PARAMS['taut'] = MD_PARAMS['taut'] * fs

print('Molecule: {:s}, Calculator: {:s}'.format(args.symbols, args.energy_type))

molecule.set_calculator(calc)

MaxwellBoltzmannDistribution(molecule, temperature_K=args.temperature)
Stationary(molecule)  # zero linear momentum
ZeroRotation(molecule)  # zero angular momentum
 
p = molecule.get_momenta()
psum = p.sum(axis=0)/float(len(p))
p = p - psum
molecule.set_momenta(p)

#filename = str(args.process_id) + '_' + MD_PARAMS['thermostat'] + '_' + args.symbols + '.traj'

md = run_molecular_dynamics(molecule, 
                            MD_PARAMS, 
                            args.n_steps, 
                            args.n_interval, 
                            trajectory_filename = filename + '.traj', 
                            return_temperature = True,
                            return_collective_variable = True)

traj = Trajectory(filename + '.traj')

temperature = torch.tensor([molecule.get_temperature() for molecule in traj]).detach()

#TODO: add saving args, add title to plot

#save_json_args(args, 'md', args.process_id, path = getcwd() + '/')

fig, ax = plt.subplots(figsize=(10, 7))

text = ''

for i, (key, value) in enumerate(MD_PARAMS.items()):
    if i == 0:
        text = '{:s}: {:s}'.format(key, str(value))
    else:
        text += '\n{:s}: {:s}'.format(key, str(value))

set_plot_sequential_data(temperature, ax=ax, window_size=100)

ax.set_xlabel('Step')
ax.set_ylabel('Temperature (K)')

ax.text(x=0.05, y=0.75, s=text, transform=ax.transAxes, fontsize=14,)

ax.set_title('Molecular dynamics simulation of {:s} with {:s} thermostat'.format(args.symbols, MD_PARAMS['thermostat']))

plt.savefig(filename + '.png')
