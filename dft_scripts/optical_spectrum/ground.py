import sys

from ase.io import read
from gpaw import GPAW, Mixer
from flonacomldft.utils.data_utils import load_from_pickle

i = int(sys.argv[1])
molecules = load_from_pickle("../molecules")

atoms = molecules[i]
atoms.set_cell([16, 16, 16])
atoms.center()

calc = GPAW(h=0.2,
            nbands=-30,
            xc='PBE',
            poissonsolver={'remove_moment': 1 + 3 + 5},
            convergence={'bands': -8},
            txt='gs_'+str(i)+'.out',
            mixer=Mixer(beta=0.1),
            eigensolver = "cg"
)

atoms.calc = calc
atoms.get_potential_energy()
calc.write('gs_'+str(i)+'.gpw', mode='all')
