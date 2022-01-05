from ase.io import read
from gpaw import GPAW
from ase.optimize import BFGS
from  ase.md.nvtberendsen import NVTBerendsen
from ase import units

mol = read('ag6.xyz')

mol.set_cell([16, 16, 16])
mol.set_pbc(True)
mol.center()

calc = GPAW(mode="lcao", h=0.2, spinpol=True, xc="PBE", symmetry="off", nbands = -4, txt='ag6.out')

mol.set_calculator(calc)

opt = BFGS(mol, trajectory='ag6.traj', logfile='qn.log')

opt.run(0.01)

