import numpy as np

from ase.io import write
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

from gpaw import GPAW

from flonacomldft.utils.silver_isomers_utils import get_molecule_isomer_minima

isomer = "ag6_planar"
mode = "lcao"

molecule = get_molecule_isomer_minima(isomer)

molecule.set_cell([16, 16, 16])
molecule.set_pbc(True)
molecule.center()

name = isomer + "_" + mode
calc = GPAW(
    mode=mode,
    h=0.2,
    spinpol=True,
    xc="PBE",
    basis="pvalence.dz",
    symmetry="off",
    nbands=-4,
    txt=name + ".out",
)

molecule.set_calculator(calc)

MaxwellBoltzmannDistribution(molecule, temperature_K=300)
Stationary(molecule)
ZeroRotation(molecule)

dyn = NVTBerendsen(
    molecule, 5 * units.fs, taut=50, temperature_K=300, trajectory=name + "_md.traj"
)
dyn.run(10)
