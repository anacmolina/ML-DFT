# TODO: Add this to package

import argparse
import json
import time
import numpy as np

from gpaw import GPAW
from ase.optimize import BFGS

from flonacomldft.utils.silver_isomers_utils import (
    isomers,
    get_molecule_isomer_minima
)

# for naming files
date_start = time.strftime('%H:%M:%S %d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('-ml', '--mode-label', type=int, default=0)
parser.add_argument('-gpwmd', '--gpaw-mode', type=str, default='lcao')
parser.add_argument('-id', '--slurm-id', type=str, default=str(random_id))

args = parser.parse_args()

args.date_start = date_start
args.random_seed = random_id

print(args)

isomer = list(isomers.keys())[args.mode_label]
mode = args.gpaw_mode

mol = get_molecule_isomer_minima(isomer)

mol.set_cell([16, 16, 16])
mol.set_pbc(True)
mol.center()

name = isomer + "_" + mode + "_" + args.slurm_id

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

mol.set_calculator(calc)
opt = BFGS(mol, trajectory = name + ".traj", logfile = name + ".log")
opt.run(0.01)

date_end = time.strftime('%H:%M:%S %d-%m-%Y')
args.date_end = date_end

argparse_dict = vars(args)

with open('args_'+id, "w") as outfile:
    json.dump(argparse_dict, outfile)