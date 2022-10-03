import sys

from gpaw import GPAW, Mixer
from flonacomldft.data_utils import load_from_pickle

i=int(sys.argv[1])

calc = GPAW('gs_'+str(i)+'.gpw')
calc = calc.fixed_density(nbands=-30,
                          convergence={'bands': "all"},
                          maxiter=1000,
                          txt='unocc_'+str(i)+'.out',
                          symmetry="off"
)

calc.write('unocc_'+str(i)+'.gpw', mode='all')
