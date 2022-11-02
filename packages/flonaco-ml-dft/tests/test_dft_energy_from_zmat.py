from flonacomldft.dft_utils import compute_energy
from flonacomldft.molecule import Ag6Isomers
from flonacomldft.internal_coordinates import get_internal_coordinates

molecule = Ag6Isomers('ag6_planar')
zmat = get_internal_coordinates([molecule])[0]
U = compute_energy(zmat)

print('Planar isomer U: {:.4}'.format(U))