# TODO: rewrite as a PyTest test
from ase.parallel import parprint as print

from datetime import datetime
import gpaw.mpi as mpi

import torch
import pandas as pd

from flonacomldft.dft_calculator import DFTCalculator
from flonacomldft.utils.silver_isomers_utils import get_molecule_isomer_minima

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

cores = mpi.world.size

isomer = get_molecule_isomer_minima('ag6_planar')

filename = "time_energy.out"
f = open(filename, "w")

dft_calculator = DFTCalculator()

startTime = datetime.now().timestamp()
txt = "ag6_cores_" + str(cores) + ".out"
pot_energy = dft_calculator.calculate_potential_energy(isomer, file_name=txt)
time_energy = datetime.now().timestamp() - startTime

print(
    "Cores:{}".format(cores),
    "Time:{:.1f}s".format(time_energy),
    "U:{:0.3f}eV".format(pot_energy),
    end="  \n",
    file=f,
)

f.close()


import numpy as np
# from flonacomldft.dft_utils import compute_energy - deprecated
from flonacomldft.utils.silver_isomers_utils import molecule
from flonacomldft.internal_coordinates import (
    get_internal_coordinates,
)

#TODO: pytest, add values like version, pbe, etc

# ag6 = get_molecule_isomer_minima('ag6_planar')
# zmat = get_internal_coordinates([ag6])[0]
# U = (zmat)

#TODO: assert etc
#assert (U)