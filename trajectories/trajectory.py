import os
import sys
import numpy as np

from ase.parallel import parprint
from ase.io import read
from gpaw import GPAW
from ase.optimize import BFGS
from  ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)

if os.path.isdir('/mnt/home/amolina/ceph/database'):
   ceph_home = '/mnt/home/amolina/ceph/database'
elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'):
   ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'
elif os.path.isdir('/home/anacristina/ml_dft_project/database/'):
    ceph_home = '/home/anacristina/ml_dft_project/database/'
elif os.path.isdir('/home/amolina/ml_dft_project/database/'):
    ceph_home = '/home/amolina/ml_dft_project/database/'
else:
   raise RuntimeError('Data path not understood')

isomer = str(sys.argv[1])
method = str(sys.argv[2])
name = isomer+"_"+method
mol = read(ceph_home + 'ag6_'+name+'.xyz')

mol.set_cell([16, 16, 16])
mol.set_pbc(True)
mol.center()

calc = GPAW(mode="lcao", h=0.2, basis="pvalence.dz", spinpol=True, xc="PBE", symmetry="off", nbands = -4, txt='ag6\
_'+name+'.out')

mol.set_calculator(calc)

MaxwellBoltzmannDistribution(mol, temperature_K=300)
Stationary(mol)
ZeroRotation(mol)

dyn = NVTBerendsen(mol, 5 * units.fs, taut = 50, temperature_K=300, trajectory='ag6_'+name+'_md.traj')
dyn.run(5000)


