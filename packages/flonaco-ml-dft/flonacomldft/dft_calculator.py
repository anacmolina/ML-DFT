import os
from ase import units
from ase.io import Trajectory
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
#from ase.parallel import parprint as print

from flonacomldft.internal_coordinates import get_internal_coordinates_from_trajectory

from gpaw import GPAW
import gpaw.mpi as mpi

class DFTCalculator:
    """
    DFT calculator class holding the GPAW calculator and the 
    calculate_potential_energy()
    """
    def __init__(self):
        super().__init__()

        self.calculator = None
        self.cell = [16, 16, 16]

    def initialize_calculator(self, filename='ag6', rank=0, path=os.getcwd()):
        
        self.path = os.path.join(path, 'DFTComputations')    
        self.file = self.path + '/' + filename + '.out'

        if rank==0 and os.path.isdir(self.path)==False:
            #print('Folder created: %s Rank: %d '%(self.path, rank))
            os.makedirs(self.path, exist_ok=True)
        else:
            pass
        
        # DFT calculator low level precision but faster (takes 1 minute in serial)                                \
        self.calculator = GPAW(mode = 'lcao', basis='pvalence.dz', h =0.2, xc = 'PBE',
                            spinpol = True, nbands = -4, txt=self.path + "/init_calc.out")

        # with higher precision, takes longer (about 30 minutes in serial).                      
        
        #self.calculator = GPAW(mode = 'fd', h =0.18, xc = 'PBE', eigensolver = 'rmm-diis', spinpol = True, nbands=-4) 

    def calculate_potential_energy(self, molecule, filename=None):
        
        if self.calculator is None:
            self.initialize_calculator()

        if filename is not None:
            self.file = self.path + '/' + filename

        self.calculator.set(txt=self.file)

        # Setting the cell parameters
        molecule.set_cell(self.cell)
        molecule.center()
        molecule.set_pbc(True)
        # print(self.calculator)
        molecule.set_calculator(self.calculator)
      
        # Calculating the potential energy
        return molecule.get_potential_energy()

    def run_molecular_dynamics(self, molecule, iters, filename, starting=True,
                               temp=300):
        rank = mpi.world.rank
        mpi.world.barrier()

        #Setting the cell
        molecule.set_cell(self.cell)
        molecule.set_pbc(True)
        molecule.center()

        # Building calculator
        if self.calculator is None:
            self.initialize_calculator(filename=filename, rank=rank)

        molecule.set_calculator(self.calculator)
    
        if starting==True:
            # Adding conditions to the MD simulation
            MaxwellBoltzmannDistribution(molecule, temperature_K=temp)
            Stationary(molecule)
            ZeroRotation(molecule)
        else:
            pass

        mpi.world.barrier()

        # Running the MD
        dyn = NVTBerendsen(molecule, 5 * units.fs, taut=50, temperature_K=temp,
                           trajectory=self.file+'.traj')
        dyn.run(iters)
        
        # Getting the MD trajectory
        mpi.world.barrier()
        traj = Trajectory(self.file+'.traj')

        return traj

# TODO: review this
def run_md_get_zmat(molecule, iterations, filename, starting=True):
    dft_calculator = DFTCalculator()
    dft_calculator.initialize_calculator(filename=filename)
    md_traj = dft_calculator.run_molecular_dynamics(molecule, iterations,
                                                filename, starting)
    zmat = get_internal_coordinates_from_trajectory(md_traj).detach()
    return zmat