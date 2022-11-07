import numpy as np
from flonacomldft.dft_utils import compute_energy
from flonacomldft.silver_isomers import molecule
from flonacomldft.internal_coordinates import (
    get_internal_coordinates,
)

#TODO: pytest, add values like version, pbe, etc

ag6 = molecule('ag6_planar')
zmat = get_internal_coordinates([ag6])[0]
U = compute_energy(zmat)

#TODO: assert etc
#assert (U)