import numpy as np
# from flonacomldft.dft_utils import compute_energy - deprecated
from flonacomldft.silver_isomers_utils import molecule
from flonacomldft.internal_coordinates import (
    get_internal_coordinates,
)

#TODO: pytest, add values like version, pbe, etc

# ag6 = get_molecule_isomer_minima('ag6_planar')
# zmat = get_internal_coordinates([ag6])[0]
# U = (zmat)

#TODO: assert etc
#assert (U)