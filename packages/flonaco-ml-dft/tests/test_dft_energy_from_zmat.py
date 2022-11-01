from flonacomldft.dft_utils import compute_energy
from flonacomldft.molecule import Ag6Isomers

#TODO: Build molecule itself in molecule
#TODO: Function to go from xyz to zmat for one molecules

molecule = Ag6Isomers('ag6_planar')
print(molecule)
U = compute_energy(molecule)

print('Planar isomer U: {:.2}'.format(U))