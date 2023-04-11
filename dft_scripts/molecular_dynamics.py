### Import modules
import argparse
import time

from gpaw import GPAW
import gpaw.mpi as mpi
from ase import units

from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)

from flonacomldft.utils.silver_isomers_utils import get_molecule_isomer_minima
from flonacomldft.utils.io_utils import get_process_id

### Get start date and process id
date_start = time.strftime('%Y-%m-%d %H:%M:%S') 
process_id = get_process_id(date_start)

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare simulation params')
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-tt', '--thermostat-type', type=str, default='berendsen')
parser.add_argument('-ns', '--n-steps', type=int, default=10)
parser.add_argument('-ts', '--time-step', type=float, default=5)
parser.add_argument('-T', '--temperature', type=float, default=300)
parser.add_argument('-taut', '--taut', type=float, default=50)
parser.add_argument('-f', '--friction', type=float, default=0.01)
parser.add_argument('-ap', '--andersen-prob', type=float, default=2e-3)
parser.add_argument('-pid', '--process-id', type=int, default=process_id)

args = parser.parse_args()
args.date_start = date_start

#mode_label = args.mode_label

### Get molecule to start simulation
mol = get_molecule_isomer_minima('is'+str(args.mode_label))

### Set simulation parameters
mol.set_cell([16, 16, 16])
mol.set_pbc(True)
mol.center()

filename = 'is{:d}_{:s}_{:d}'.format(args.mode_label, args.thermostat_type, args.process_id)

### Set calculator
calc = GPAW(mode="lcao", h=0.2, basis="pvalence.dz", spinpol=True, xc="PBE", symmetry="off", txt = filename +'.out')
mol.calc = calc

### Set initial conditions
MaxwellBoltzmannDistribution(mol, temperature_K=300)
Stationary(mol)
ZeroRotation(mol)

params_thermostat = {'time_step': args.time_step,
                     'temperature': args.temperature,
                     'taut': args.taut,
                     'friction': args.friction,
                     'andersen_prob': args.andersen_prob,
                     'filename': filename + '.traj'}

print(params_thermostat)
print(args)

### Set thermostat
def set_thermostat(thermostat_type, molecule, params):
    
    if thermostat_type == 'berendsen':
        from ase.md.nvtberendsen import NVTBerendsen
        thermostat = NVTBerendsen(molecule, 
                           params['time_step'] * units.fs, 
                           taut = params['taut'], 
                           temperature_K = params['temperature'], 
                           trajectory = params['filename'])
        
        params_used = {'thermostat_type': 'Berendsen',
                        'time_step': params['time_step'],
                        'temperature': params['temperature'],
                        'taut': params['taut']}
    
    elif thermostat_type == 'langevin':
        from ase.md.langevin import Langevin
        thermostat = Langevin(molecule,
                              params['time_step'] * units.fs,
                              friction = params['friction'],
                              temperature_K = params['temperature'],
                              trajectory = params['filename'])
        
        params_used = {'thermostat_type': 'Langevin',
                        'time_step': params['time_step'],
                        'temperature': params['temperature'],
                        'friction': params['friction']}
        
    elif thermostat_type == 'andersen':
        from ase.md.andersen import Andersen
        thermostat = Andersen(molecule,
                              params['time_step'] * units.fs,
                              andersen_prob = params['andersen_prob'],
                              temperature_K = params['temperature'],
                              trajectory = params['filename'])
        
        params_used = {'thermostat_type': 'Andersen',
                        'time_step': params['time_step'],
                        'temperature': params['temperature'],
                        'andersen_prob': params['andersen_prob']}

    else:
        raise ValueError('Thermostat type not recognized')
    
    return thermostat, params_used

### Run simulation
dyn, params_used = set_thermostat(args.thermostat_type, mol, params_thermostat)
print(params_used)
dyn.run(args.n_steps)

mpi.world.barrier()

### Get end date
date_end = time.strftime('%Y-%m-%d %H:%M:%S')
args.date_end = date_end

argparse_dict = vars(args)

mpi.world.barrier()

### Save simulation parameters
from flonacomldft.utils.io_utils import save_json_args
from os import getcwd
save_json_args(args, 'md', args.process_id, path = getcwd() + '/')

mpi.world.barrier()

### Plot temperature
from ase.io.trajectory import Trajectory
traj = Trajectory(filename + '.traj', 'r')

temp = [config.get_temperature() for config in traj]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(range(0, len(temp))), temp, '.-', label='is{:d}'.format(args.mode_label))
ax.set_title(params_used)
ax.set_xlabel('time step')
ax.set_ylabel('temperature')
ax.legend()

plt.savefig('temperature_{:d}.png'.format(args.process_id))