### Import modules
import os
from ase import units
from ase.io import Trajectory
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
#from ase.parallel import parprint as print

class EMTCalculator:
    def __init__(self):
        super().__init__()

        self.calculator = None
        self.cell = [5, 5, 5]
    
        from ase.calculators.emt import EMT
        self.calculator = EMT()

    def calculate_potential_energy(self, atoms):
        return self.calculator.get_potential_energy(atoms)

class DFTCalculator:
    """
    DFT calculator class holding the GPAW calculator and the 
    calculate_potential_energy()
    """
    def __init__(self):
        super().__init__()

        self.calculator = None
        self.cell = [16, 16, 16]

    def initialize_calculator(self, params=None, predefined_params='LCAO', foldername='DFTComputations', filename='ag6.out', path=os.getcwd()):
        
        self.path = os.path.join(path, foldername)    
        self.file = self.path + '/' + filename

        from gpaw import GPAW
        
        #import gpaw.mpi as mpi
        #rank = mpi.world.rank
        #if rank==0 and os.path.isdir(self.path)==False:

        if os.path.isdir(self.path)==False:
            #print('Folder created: %s Rank: %d '%(self.path, rank))
            os.makedirs(self.path, exist_ok=True)
        else:
            pass

        #mpi.world.barrier()

        if params is None:
            from flonacomldft.utils.silver_isomers_utils import get_molecule_calc_params
            params = get_molecule_calc_params(predefined_params)

        if 'txt' not in params:
            params['txt'] = self.path + '/init_calc.out'
        
        #DFT calculator low level precision but faster (takes 1 minute in serial)                                \
        #self.calculator = GPAW(mode = 'lcao', basis='pvalence.dz', h =0.2, xc = 'PBE', spinpol = True, nbands = -4, txt=self.path + "/init_calc.out")

        self.calculator = GPAW(**params)

        #with higher precision, takes longer (about 30 minutes in serial).                      
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


class Thermostats:

    def __init__(self):
        pass

    def __call__(self, params, atoms):

        params = self._select_params(params)
        params['atoms'] = atoms
        thermostat_name = params['thermostat']
        params.pop('thermostat')
        
        return self._thermostat(thermostat_name)(**params)

    def _select_params(self, params):

        thermostat_name = params['thermostat']
        selected_params = ['thermostat', 'timestep', 'temperature_K']

        if thermostat_name == 'berendsen':
            selected_params = selected_params + ['taut']        

        elif thermostat_name == 'langevin':
            selected_params = selected_params + ['friction']

        elif thermostat_name == 'andersen':
            selected_params = selected_params + ['andersen_prob']

        else:
            raise ValueError('Thermostat name not recognized')

        return {param: params[param] for param in selected_params} 

    def _thermostat(self, thermostat_name):

        if thermostat_name == 'berendsen':
            from ase.md.nvtberendsen import NVTBerendsen
            return NVTBerendsen

        elif thermostat_name == 'langevin':
            from ase.md.langevin import Langevin
            return Langevin

        elif thermostat_name == 'andersen':
            from ase.md.andersen import Andersen
            return Andersen

        else:
            raise ValueError('Thermostat name not recognized')


def run_molecular_dynamics(atoms, thermostat_params, n_steps, interval, trajectory_filename, return_temperature=True):

    from ase.io.trajectory import Trajectory

    dyn = Thermostats()(thermostat_params, atoms)

    print('Running molecular dynamics with {} thermostat'.format(thermostat_params['thermostat']))

    traj = Trajectory(trajectory_filename, 'w', atoms)

    if return_temperature:

        def print_temperature(a=atoms):
            temperature = a.get_temperature()
            print('Temperature: {:.1f} K'.format(temperature))

        dyn.attach(print_temperature, interval=interval)

    dyn.attach(traj.write, interval=interval)

    dyn.run(n_steps)

    return dyn


# TODO: Delete this function
#     def run_molecular_dynamics(self, molecule, iters, filename, starting=True,
#                                temp=300):
#         rank = mpi.world.rank
#         mpi.world.barrier()
# 
#         #Setting the cell
#         molecule.set_cell(self.cell)
#         molecule.set_pbc(True)
#         molecule.center()
# 
#         # Building calculator
#         if self.calculator is None:
#             self.initialize_calculator(filename=filename, rank=rank)
# 
#         molecule.set_calculator(self.calculator)
#     
#         if starting==True:
#             # Adding conditions to the MD simulation
#             MaxwellBoltzmannDistribution(molecule, temperature_K=temp)
#             Stationary(molecule)
#             ZeroRotation(molecule)
#         else:
#             pass
# 
#         mpi.world.barrier()
# 
#         # Running the MD
#         dyn = NVTBerendsen(molecule, 5 * units.fs, taut=50, temperature_K=temp,
#                            trajectory=self.file+'.traj')
#         dyn.run(iters)
#         
#         # Getting the MD trajectory
#         mpi.world.barrier()
#         traj = Trajectory(self.file+'.traj')
# 
#         return traj


#TODO: Add optimization and optical spectrum methods