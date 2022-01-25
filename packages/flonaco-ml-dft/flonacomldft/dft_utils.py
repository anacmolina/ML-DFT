import chemcoord as cc
import torch
from gpaw import GPAW
import numpy as np

# Transformation for angles 
class Angles_tranformation(torch.Tensor):
    def __init__(self, x_):
        super().__init__()
        self.x = x_
        self.n = 6

    def transf(self, n_=None):
        if n_!=None:
            self.n = n_
        self.x[self.n:-1] = torch.tanh(self.x[self.n:])

    def inv_transf(self, n_=None):
        if n_!=None:
            self.n = n_
        self.x[self.n:-1] = torch.arctanh(self.x[self.n:])

# Calculating the energy

class Structure:
    def __init__(self, construction_table_, symbols_, Natoms_):
        super().__init__()
        self.zmat_values = None
        self.construction_table = construction_table_.copy()
        self.symbols = symbols_
        self.Natoms = Natoms_
        self.zmat_matrix = None
        self.molecule = None
        self.potential_energy = None
        self.calculator = None

    def build_zmat_matrix(self, zmat_values_=None):
        if(zmat_values_!=None):
            self.zmat_values = zmat_values_
        
        zmat_values = self.zmat_values.reshape(3, self.Natoms)
        zmat_values[1:] = np.rad2deg(zmat_values[1:])

        zmat_matrix = self.construction_table.copy()
       
        zmat_matrix.insert(0, "atom", self.symbols, True)  
        zmat_matrix.insert(2, "bond", zmat_values[0], True)
        zmat_matrix.insert(4, "angle", zmat_values[1], True)
        zmat_matrix.insert(6, "dihedral", zmat_values[2], True)

        self.zmat_matrix = cc.Zmat(zmat_matrix)
        self.molecule = self.zmat_matrix.get_cartesian().get_ase_atoms()

    def calculate_potential_energy(self, zmat_values_):
        self.zmat_values = zmat_values_

        if (self.molecule == None):
            self.build_zmat_matrix()
        
        cell = [16, 16, 16]
        self.molecule.set_cell(cell)
        self.molecule.center()
        self.molecule.set_pbc(True)

        # DFT calculator low level precision but faster (takes 1 minute in serial)
        self.calculator = GPAW(mode = 'lcao', h =0.2, xc = 'PBE', spinpol = True, nbands = -4)

        #DFT calculator with higher precision but takes longer (about 30 minutes in serial).
        #calc = GPAW(mode = 'fd', h =0.18, xc = 'PBE', eigensolver = 'rmm-diis', spinpol = True, nbands=-4)

        self.molecule.set_calculator(self.calculator)
        
        self.potential_energy = self.molecule.get_potential_energy()
        
