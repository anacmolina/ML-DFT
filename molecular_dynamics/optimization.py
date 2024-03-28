# libraries
import argparse
import time
import numpy as np

from gpaw import GPAW
from ase.optimize import BFGS

from abflowmc.utils.silver_isomers_utils import (
    isomers,
    get_molecule_isomer_minima,
    get_process_id
)
from abflowmc.utils.io_utils import save_json_args

# get start time and process id
date_start = time.strftime('%Y-%m-%d %H:%M:%S')
process_id = get_process_id(date_start)

# define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-gpwmd', '--gpaw-mode', type=str, default='lcao')
parser.add_argument('-id', '--process-id', type=str, default=str(process_id))

args = parser.parse_args()
args.date_start = date_start

print(args)

# get molecule
molecule = get_molecule_isomer_minima('is{:d}'.format(args.mode_label))

# set mode calculation and parameters
mode = args.gpaw_mode

molecule.set_cell([16, 16, 16])
molecule.set_pbc(True)
molecule.center()

name = "is{:d}_{:s}_{:d}".format(args.mode_label, args.gpaw_mode, args.process_id)

# set calculator
calc = GPAW(
    mode=args.gpaw_mode,
    h=0.2,
    spinpol=True,
    xc="PBE",
    basis="pvalence.dz",
    symmetry="off",
    nbands=-4,
    txt=name + ".out",
)

molecule.calc = calc

# run optimization
opt = BFGS(molecule, trajectory = name + ".traj", logfile = name + ".log")
opt.run(0.01)

# get end time
date_end = time.strftime('%Y-%m-%d %H:%M:%S')
args.date_end = date_end

args.algorithm = 'optimization.py'

# save minimization parameters
save_json_args(args)
