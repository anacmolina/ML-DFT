# load libraries
from os import getcwd

import argparse
import time
from ase.parallel import parprint as print

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

from abflowmc.utils.silver_isomers_utils import (
    get_molecule_isomer_minima,
    get_molecule_calc_params,
)
from abflowmc.utils.io_utils import save_json_args

from abflowmc.dft_calculator import run_molecular_dynamics

from abflowmc.utils.io_utils import set_str_date_to_int

from abflowmc.utils.plots import set_plot_sequential_data

# load parallel libraries

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

comm.broadcast(num_seed, 0)
comm.broadcast(date_start, 0)

num_seed = num_seed[0]
date_start = date_start[0]

print('Num_seed: {:d}, Date_start: {:d}'.format(num_seed, date_start))

# set up parser 

parser = argparse.ArgumentParser(description='Run molecular dynamics simulation of silver isomers')
# system parameters
parser.add_argument('-isomer', '--isomer-label' , type=int, default=0, help='Isomer label')
parser.add_argument('-etype', '--energy-type', type=str, default='dft', help='Energy type')
# molecular dynamics parameters
parser.add_argument('-tt', '--thermostat-type', type=str, default='berendsen')
parser.add_argument('-ns', '--n-steps', type=int, default=5)
parser.add_argument('-ts', '--time-step', type=float, default=5)
parser.add_argument('-nin', '--n-interval', type=int, default=1)
parser.add_argument('-T', '--temperature', type=float, default=300)
parser.add_argument('-taut', '--taut', type=float, default=None) #50
parser.add_argument('-f', '--friction', type=float, default=None) #2e-3
parser.add_argument('-ap', '--andersen-prob', type=float, default=None) #2e-3
parser.add_argument('-pid', '--process-id', type=int, default=date_start)
parser.add_argument('-s', '--seed', type=int, default=num_seed)

args = parser.parse_args()
args.date_start = date_start

# define isomer and calculator

if args.energy_type == 'dft':
    
    MOLECULE = 'is{:d}'.format(args.isomer_label)

    params_calc = get_molecule_calc_params()
    params_calc['txt'] = '{:d}_is{:d}_{:s}.out'.format(args.process_id, args.isomer_label, args.thermostat_type)
    calc = GPAW(**params_calc)

if args.energy_type == 'emt':
    
    from ase.calculators.emt import EMT
    
    MOLECULE = 'emt_is{:d}'.format(args.isomer_label)
    calc = EMT()

TEMPERATURE = args.temperature
N_STEPS = args.n_steps
INTERVAL = args.n_interval

MD_PARAMS = {'thermostat': args.thermostat_type,
             'timestep': args.time_step * fs,
             'temperature_K': TEMPERATURE,
             'taut': args.taut,
             'andersen_prob': args.andersen_prob,
             'friction': args.friction,
}

if MD_PARAMS['thermostat'] == 'berendsen':
    MD_PARAMS['taut'] = MD_PARAMS['taut'] * fs

print('Molecule: {:s}, Calculator: {:s}'.format(MOLECULE, args.energy_type))

molecule = get_molecule_isomer_minima(MOLECULE)

molecule.set_cell([16, 16, 16])
molecule.set_pbc(True)
molecule.center()

molecule.set_calculator(calc)

MaxwellBoltzmannDistribution(molecule, temperature_K=TEMPERATURE)
Stationary(molecule)  # zero linear momentum
ZeroRotation(molecule)  # zero angular momentum
 
p = molecule.get_momenta()
psum = p.sum(axis=0)/float(len(p))
p = p - psum
molecule.set_momenta(p)

filename = str(args.process_id) + '_' + MD_PARAMS['thermostat'] + '_' + MOLECULE + '.traj'

md = run_molecular_dynamics(molecule, 
                            MD_PARAMS, 
                            N_STEPS, 
                            INTERVAL, 
                            trajectory_filename=filename, 
                            return_temperature=True,
                            return_collective_variable=True)

traj = Trajectory(filename)

temperature = torch.tensor([molecule.get_temperature() for molecule in traj]).detach()

save_json_args(args, 'md', args.process_id, path = getcwd() + '/')

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

ax.set_title('Molecular dynamics simulation of {:s} with {:s} thermostat'.format(MOLECULE, MD_PARAMS['thermostat']))

plt.savefig('{:d}_{:s}_{:s}.png'.format(args.process_id, MD_PARAMS['thermostat'], MOLECULE))
