import sys
import numpy as np
from ase import Atoms
from gpaw import GPAW
from ase.optimize import BFGS

mol_is1 = Atoms('Ag6', positions=np.array([[ 7.98046 ,  5.464791,  8.      ],
                                       [ 7.961142, 10.16485 ,  8.      ],
                                       [ 6.58375 ,  7.868667,  8.      ],
                                       [ 5.299389,  5.513795,  8.      ],
                                       [ 9.369786,  7.87624 ,  8.      ],
                                       [10.665305,  5.524828,  8.      ]]))

mol_is2 = Atoms('Ag6', positions=np.array([[ 6.591008,  5.595878,  7.13902 ],
                                       [ 9.408992,  5.595878,  7.13902 ],
                                       [ 5.715757,  8.27292 ,  7.161092],
                                       [ 8.      ,  7.529808,  8.358414],
                                       [10.284243,  8.27292 ,  7.161092],
                                       [ 8.      ,  9.919833,  7.129457]]))

isomer = str(sys.argv[1])
method = str(sys.argv[2])

mol = None

if (isomer=="is1"):
    mol = mol_is1
elif (isomer=="is2"): 
    mol = mol_is2
else:
    raise RuntimeError("Set a define isomer.")
    
mol.set_cell([16, 16, 16])
mol.set_pbc(True)
mol.center()

calc = None

name = isomer+"_"+method
if (method=="fd"):
    calc_fd = GPAW(mode="fd", h=0.18, spinpol=True, xc="PBE", symmetry="off", nbands = -4,
                   txt="ag6_"+name+".out")
    calc = calc_fd
elif (method=="lcao"):
    calc_lcao = GPAW(mode="lcao", h=0.2, basis="pvalence.dz", spinpol=True, xc="PBE", symmetry="off",
                     nbands = -4, txt="ag6_"+name+".out")
    calc = calc_lcao
else:
    raise RuntimeError("Set a define method.")

mol.set_calculator(calc)
opt = BFGS(mol, trajectory="ag6_"+name+"_opt.traj", logfile="qn_"+name+".log")
opt.run(0.01)

write("ag6_"+name+".xyz", mol)


