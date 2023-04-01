import argparse
import time

from gpaw import GPAW
import gpaw.mpi as mpi
from ase import units

from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.andersen import Andersen

from flonacomldft.utils.silver_isomers_utils import get_molecule_isomer_minima
from flonacomldft.utils.io_utils import get_project_path
from flonacomldft.utils.silver_isomers_utils import isomers

from flonacomldft.utils.io_utils import get_date_process_id

print('ranks: ', mpi.world.size)

# for naming files
date_start, process_id = get_date_process_id()

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare simulation params')
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-ni', '--n-iter', type=int, default=10)
parser.add_argument('-ts', '--time-step', type=float, default=5)
parser.add_argument('-ap', '--andersen-prob', type=float, default=2e-3)
parser.add_argument('-pid', '--process-id', type=int, default=process_id)

args = parser.parse_args()
args.date_start = date_start

path = get_project_path() + 'ag6_dft_calculations/md_trajectories/'

mode_label = args.mode_label

mol = get_molecule_isomer_minima('is'+str(mode_label))

mol.set_cell([16, 16, 16])
mol.set_pbc(True)
mol.center()

calc = GPAW(mode="lcao", h=0.2, basis="pvalence.dz", spinpol=True, xc="PBE", symmetry="off", txt=path+'ag6_is{:d}_{:d}.out'.format(args.mode_label, args.process_id))

mol.calc = calc
#opt = BFGS(mol, trajectory=path+'ag6_opt_is{:d}_{:d}.traj'.format(args.mode_label, args.process_id), logfile=path+'qn_{:d}.log'.format(args.process_id))
#opt.run(0.01)

MaxwellBoltzmannDistribution(mol, temperature_K=300)
Stationary(mol)
ZeroRotation(mol)

dyn = NVTBerendsen(mol, args.time_step * units.fs, taut = 50, temperature_K=300, trajectory=path+'ag6_berendsen_md_is{:d}_{:d}.traj'.format(args.mode_label, args.process_id))
dyn.run(5000)

mpi.world.barrier()

from ase.io.trajectory import Trajectory

mol = Trajectory(path + 'ag6_berendsen_md_is{:d}_{:d}.traj'.format(args.mode_label, args.process_id))[-1]
mol.calc = calc

dyn = Andersen(mol, args.time_step * units.fs, temperature_K=300, andersen_prob=args.andersen_prob, trajectory=path+'ag6_andersen_md_is{:d}_{:d}.traj'.format(args.mode_label, args.process_id))
dyn.run(args.n_iter)

date_end = time.strftime('%H:%M:%S %d-%m-%Y')
args.date_end = date_end

argparse_dict = vars(args)

mpi.world.barrier()

from flonacomldft.utils.io_utils import save_json_args
save_json_args(args, 'md', args.process_id, path)

mpi.world.barrier()

import matplotlib.pyplot as plt

traj_berendsen = Trajectory(path + 'ag6_berendsen_md_is{:d}_{:d}.traj'.format(args.mode_label, args.process_id))
traj_andersen = Trajectory(path + 'ag6_andersen_md_is{:d}_{:d}.traj'.format(args.mode_label, args.process_id))

temp_berendsen = [config.get_temperature() for config in traj_berendsen]
temp_andersen = [config.get_temperature() for config in traj_andersen]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(range(0, len(temp_berendsen))), temp_berendsen, '.-', label='Berendsen')
ax.plot(list(range(len(temp_berendsen), len(temp_berendsen)+len(temp_andersen))), temp_andersen, '.-', label='Andersen')
ax.set_title('prob: T = {:.1e}'.format(args.andersen_prob) )
ax.set_xlabel('time step')
ax.set_ylabel('temperature')
ax.legend()

plt.savefig('temperature_{:d}.png'.format(args.process_id))