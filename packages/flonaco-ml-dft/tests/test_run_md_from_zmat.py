import numpy as np
from flonacomldft.dft_utils import (run_molecular_dynamics)
from flonacomldft.silver_isomers import molecule
from flonacomldft.internal_coordinates import (
    get_internal_coordinates,
)

ag6 = molecule('ag6_planar')
U = run_molecular_dynamics(ag6, 5, 'test')
