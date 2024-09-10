"""
Description: This script runs a molecular dynamics simulation.
"""

# import libraries
import datetime
import argparse
import logging

from ase.units import kB, fs
from gpaw import GPAW
import gpaw.mpi as mpi
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

from flonacomldft.utils.io_utils import set_str_date_to_int
from flonacomldft.utils.silver_isomers_utils import (
    get_molecule_isomer_minima,
    get_calculator_params
)
from flonacomldft.dft_calculator import run_molecular_dynamics

#set up arguments
def parse_args():

    start_time = datetime.datetime.now()
    id_number = set_str_date_to_int(str(start_time).split('.')[0])

    parser = argparse.ArgumentParser(description="Run molecular dynamics simulation")

    parser.add_argument('-symbols', '--symbols', type=str, help='Symbols of the molecule')
    parser.add_argument('-isomer', '--isomer-label', type=int, help='Isomer label')
    parser.add_argument('-cell', '--cell', type=float, help='Cell size')
    parser.add_argument('-vacuum', '--vacuum', type=float, help='Vacuum size')
    parser.add_argument('-pbc', '--pbc', type=bool, default=True, help='Periodic boundary conditions')
    parser.add_argument('-gpwmd', '--gpaw-mode', type=str, default='LCAO', help='GPAW mode')
    parser.add_argument('-etype', '--energy-type', type=str, default='dft', help='Energy type')
    parser.add_argument('-tt', '--thermostat-type', type=str, help='Thermostat type')
    parser.add_argument('-ns', '--nsteps', type=int, default=100, help='Number of steps')
    parser.add_argument('-ts', '--time-step', type=float, default=5)
    parser.add_argument('-ninterval', '--n-interval', type=int, default=1)
    parser.add_argument('-T', '--temperature', type=float, default=300)
    parser.add_argument('-taut', '--taut', type=float, default=None) #50
    parser.add_argument('-f', '--friction', type=float, default=None) #2e-3
    parser.add_argument('-ap', '--andersen-prob', type=float, default=None) #2e-3
    
    parser.add_argument('-id', '--process-id', type=int, default=id_number, help='ID of the process (start time by default)')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Seed for random number generator')
    
    args = parser.parse_args()

    filename = "molecular_dynamics_{:s}_{:d}_{:d}".format(args.symbols, args.isomer_label, args.process_id)

    args.filename = filename

    return args, start_time

# set up logger
def init_logger():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'  # Time format
    )

    logger = logging.getLogger()

    return logger

# set cell or vacuum
def set_cell_or_vacuum(molecule, cell, vacuum):
    
    if cell is not None and vacuum is None:

        molecule.set_cell([cell, cell, cell])
        molecule.center()

    elif cell is None and vacuum is not None:

        molecule.center(vacuum=vacuum)

    else:
        
        raise ValueError("Cell or vacuum cannot be set at the same time")

    return molecule

# set molecular dynamics parameters
def get_molecular_dynamics_params(args):

    md_params = {'thermostat': args.thermostat_type,
             'timestep': args.time_step * fs,
             'temperature_K': args.temperature,
             'taut': args.taut,
             'andersen_prob': args.andersen_prob,
             'friction': args.friction,
    }

    if md_params['thermostat'] == 'berendsen':
        md_params['taut'] = md_params['taut'] * fs

    return md_params

def main():

    args, start_time = parse_args()
    filename = args.filename

    logger = init_logger()

    if mpi.rank == 0:

        logger.addHandler(logging.FileHandler(f"{args.filename}.log"))
        logger.info("MOLECULAR DYNAMICS")

        for key, value in vars(args).items():
            if key != "filename" and key != "start_time":
                logger.info(f"{key}: {value}")

    molecule = get_molecule_isomer_minima(args.gpaw_mode,
        "{:s}".format(args.symbols), 
        "{:d}".format(args.isomer_label)
    )

    molecule = set_cell_or_vacuum(molecule, 
                                  args.cell, 
                                  args.vacuum, 
                                  )

    molecule.set_pbc(args.pbc)

    params_calc = get_calculator_params(name=args.gpaw_mode)
    params_calc["txt"] = filename + ".out"

    calc = GPAW(**params_calc)

    molecule.calc = calc

    MaxwellBoltzmannDistribution(molecule, temperature_K=args.temperature)
    Stationary(molecule)  # zero linear momentum
    ZeroRotation(molecule)  # zero angular momentum
    
    p = molecule.get_momenta()
    psum = p.sum(axis=0)/float(len(p))
    p = p - psum
    molecule.set_momenta(p)

    md_params = get_molecular_dynamics_params(args)

    md = run_molecular_dynamics(molecule, 
            md_params, 
            args.nsteps, 
            args.n_interval, 
            trajectory_filename = filename + '.traj', 
            return_temperature = True,
            return_collective_variable = True)
    
    end_time = datetime.datetime.now()

    if mpi.rank == 0:
        logger.info(
            "simulation started at {:s}".format(str(start_time).split('.')[0])
        )
        logger.info(
            "simulation finished at {:s}".format(str(end_time).split('.')[0])
        )
        logger.info(
            "simulation took {:f} seconds".format(
                (end_time - start_time).total_seconds(
                )
            )
        )

if __name__ == '__main__':
    main()


