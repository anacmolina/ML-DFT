import numpy as np

from gpaw import GPAW
from ase.optimize import BFGS
from ase.io import write

from flonacomldft.utils.silver_isomers_utils import get_molecule_isomer_minima


isomer = "ag6_planar"
mode = "lcao"

mol = get_molecule_isomer_minima(isomer)

mol.set_cell([16, 16, 16])
mol.set_pbc(True)
mol.center()

name = isomer + "_" + mode
calc = GPAW(
    mode=mode,
    h=0.18,
    spinpol=True,
    xc="PBE",
    basis="pvalence.dz",
    symmetry="off",
    nbands=-4,
    txt=name + ".out",
)

mol.set_calculator(calc)
opt = BFGS(mol, trajectory=name + ".traj", logfile=name + ".log")
opt.run(0.01)
