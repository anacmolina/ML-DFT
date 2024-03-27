# libraries

import os
from ase import units
from ase.io import Trajectory
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)

from flonacomldft.collective_variables import compute_R

class EMTCalculator:
    """
    Class to build the EMT calculator
    """

    def __init__(self, cell=[5, 5, 5]):
        """
        Initialize the EMT calculator object
        Args:
            cell: cell size
        """

        super().__init__()

        self.calculator = None
        self.cell = cell
    
        from ase.calculators.emt import EMT
        self.calculator = EMT()

    def calculate_potential_energy(self, atoms):
        """
        Calculate the potential energy of the system
        Args:
            atoms: ASE atoms object
        Returns:
            The potential energy of the system
        """
        atoms.set_cell(self.cell)
        atoms.center()
        atoms.set_pbc(True)

        return self.calculator.get_potential_energy(atoms)

class DFTCalculator:
    """
    Class to build the DFT calculator
    """
    def __init__(self, cell=[16, 16, 16]):

        """
        Initialize the DFT calculator object
        
        Args:
            cell: cell size
        """

        super().__init__()

        self.calculator = None
        self.cell = cell

    def initialize_calculator(self, 
                              params=None, 
                              predefined_params='LCAO', 
                              foldername='DFTComputations', 
                              filename='ag6.out', 
                              path=os.getcwd()):
        
        """
        Initialize the DFT calculator
        Args:
            params: dictionary with the parameters of the calculator
            predefined_params: predefined parameters of the calculator
            foldername: name of the folder to save the output
            filename: name of the output file
            path: path to save the output
        """

        self.path = os.path.join(path, foldername)    
        self.file = self.path + '/' + filename

        from gpaw import GPAW

        if os.path.isdir(self.path)==False:
            os.makedirs(self.path, exist_ok=True)
        else:
            pass

        if params is None:
            from flonacomldft.utils.silver_isomers_utils import get_molecule_calc_params
            params = get_molecule_calc_params(predefined_params)

        if 'txt' not in params:
            params['txt'] = self.path + '/init_calc.out'
        
        self.calculator = GPAW(**params)

    def calculate_potential_energy(self, atoms, filename=None):

        """
        Calculate the potential energy of the system
        Args:
            atoms: ASE atoms object
            filename: name of the output file
        Returns:
            The potential energy of the system
        """
        
        if self.calculator is None:
            self.initialize_calculator()

        if filename is not None:
            self.file = self.path + '/' + filename

        self.calculator.set(txt=self.file)

        # set the cell parameters
        atoms.set_cell(self.cell)
        atoms.center()
        atoms.set_pbc(True)

        atoms.set_calculator(self.calculator)
      
        # calculate the potential energy
        return atoms.get_potential_energy()


class Thermostats:
    """
    Class to build the thermostat for the molecular dynamics simulation
    """

    def __call__(self, params, atoms):

        """
        Function to obtain the thermostat object for the molecular dynamics
        
        Args:
            params: dictionary with the parameters of the thermostat
            
                ex_params = {'thermostat': thermostat_type,
                 'timestep': time_step * fs,
                 'temperature_K': temperature,
                 'taut': taut,
                 'andersen_prob': andersen_prob,
                 'friction': friction
                }
            
            atoms: ASE atoms object
        
        Returns:
            The thermostat object according to params
        """

        params = self._select_params(params)
        params['atoms'] = atoms
        thermostat_name = params['thermostat']
        params.pop('thermostat')
        
        return self._thermostat(thermostat_name)(**params)

    def _select_params(self, params):

        """
        Function to select the parameters of the thermostat according to the thermostat type

        Args:
            params: dictionary with the parameters of the thermostat

        Returns:
            Dictionary with the selected parameters
        """

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

        """
        Get thermostat object according to the thermostat name

        Args:
            thermostat_name: name of the thermostat

        Returns:
            The thermostat object
        """

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


def run_molecular_dynamics(atoms, 
                           thermostat_params, 
                           n_steps, 
                           interval, 
                           trajectory_filename, 
                           return_temperature=True, 
                           return_collective_variable=True):
    
    """
    Function to run the molecular dynamics simulation
    
    Args:
        atoms: ASE atoms object
        thermostat_params: dictionary with the parameters of the thermostat
        
            ex_params = {'thermostat': thermostat_type,
             'timestep': time_step * fs,
             'temperature_K': temperature,
             'taut': taut,
             'andersen_prob': andersen_prob,
             'friction': friction
            }
        
        n_steps: number of steps of the simulation
        interval: interval to print the temperature and collective variable and save step
        trajectory_filename: name of the trajectory file
        return_temperature: boolean to return the temperature
        return_collective_variable: boolean to return the collective variable
    
    Returns:
        The molecular dynamics trajectory
    """

    from ase.parallel import parprint as print
    from ase.io.trajectory import Trajectory

    dyn = Thermostats()(thermostat_params, atoms)

    print('Running molecular dynamics with {} thermostat'.format(thermostat_params['thermostat']))

    traj = Trajectory(trajectory_filename, 'w', atoms)

    if return_temperature:

        def print_temperature(a=atoms):
            temperature = a.get_temperature()
            print('Temperature: {:.1f} K'.format(temperature))

        dyn.attach(print_temperature, interval=interval)

    if return_collective_variable:

        def print_collective_variable(a=atoms):
            R = compute_R(a)
            print('Radius of gyration: {:.2f} K'.format(R))

        dyn.attach(print_collective_variable, interval=interval)

    dyn.attach(traj.write, interval=interval)

    dyn.run(n_steps)

    return dyn
