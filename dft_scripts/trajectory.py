import os
import numpy as np

from flonacomldft.utils.silver_isomers_utils import get_molecule_isomer_minima

from ase.parallel import parprint
from gpaw import GPAW
from ase.optimize import BFGS
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

if os.path.isdir("/mnt/home/amolina/ceph/database"):
    ceph_home = "/mnt/home/amolina/ceph/database"
elif os.path.isdir("/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/"):
    ceph_home = "/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/"
elif os.path.isdir("/home/anacristina/ml_dft_project/database/"):
    ceph_home = "/home/anacristina/ml_dft_project/database/"
elif os.path.isdir("/home/amolina/ml_dft_project/database/"):
    ceph_home = "/home/amolina/ml_dft_project/database/"
else:
    raise RuntimeError("Data path not understood")

isomer = "ag6_planar"
mode = "lcao"
name = isomer + "_" + mode

molecule = get_molecule_isomer_minima(isomer)

molecule.set_cell([16, 16, 16])
molecule.set_pbc(True)
molecule.center()

calc = GPAW(
    mode=mode,
    h=0.2,
    basis="pvalence.dz",
    spinpol=True,
    xc="PBE",
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
dyn.run(5000)
