# TODO: add description
# TODO: add logging file

# import libraries
import argparse
import time
import numpy as np

from gpaw import GPAW
from ase.optimize import BFGS

from flonacomldft.utils.silver_isomers_utils import (
    get_molecule_isomer_minima,
    get_molecule_calc_params
)

from ase.parallel import parprint as print

# get start time
date_start = time.strftime('%Y-%m-%d %H:%M:%S')

# define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-symbs', '--symbols', type=str, help='Symbols of the molecule')
parser.add_argument('-isomer', '--isomer-label', type=str, help='Isomer label')
parser.add_argument('-cell', '--cell', type=float, help='Cell size')
parser.add_argument('-vacuum', '--vacuum', type=float, help='Vacuum size')
parser.add_argument('-pbc', '--pbc', type=bool, default=True, help='Periodic boundary conditions')
parser.add_argument('-gpwmd', '--gpaw-mode', type=str, default='LCAO', help='GPAW mode')
parser.add_argument('-id', '--process-id', type=str, default=date_start, help='Start time')

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

# set mode energy calculation
mode = args.gpaw_mode

# set filename
filename = "{:s}_{:s} {:s} {:s}".format(args.symbols, args.isomer_label, args.gpaw_mode.lower(), args.process_id)

# set calculator
params_calc = get_molecule_calc_params(name=args.gpaw_mode)
params_calc['txt'] = filename + ".out"

calc = GPAW(**params_calc)

molecule.calc = calc

# run optimization
opt = BFGS(molecule, 
           trajectory = filename + ".traj", 
           logfile = filename + ".log")
opt.run(0.01)

# get end time
date_end = time.strftime("%Y-%m-%d %H:%M:%S")
args.date_end = date_end

print("Start time: {:s}".format(args.date_start))
print("End time: {:s}".format(args.date_end))

#args.algorithm = 'optimization.py'
#
#### Save minimization parameters
#save_json_args(args)
