"""
Description: This script optimize the parameters for the molecule
""" 

# import libraries
import datetime
import argparse
import logging

import numpy as np

from ase import Atoms
from gpaw import GPAW, mpi
from flonacomldft.utils.io_utils import (
    set_str_date_to_int,
    init_logger
)
from flonacomldft.utils.silver_isomers_utils import (
    get_molecule_isomer_minima,
    get_calculator_params
)

# set up arguments
def parse_args():

    start_time = datetime.datetime.now()
    id_number = set_str_date_to_int(str(start_time).split('.')[0])

    parser = argparse.ArgumentParser(description="Optimization of silver isomers")
    parser.add_argument(
        "-symbols", "--symbols", type=str, help="Symbols of the molecule"
    )
    parser.add_argument("-isomer", "--isomer-label", type=int, help="Isomer label")
    parser.add_argument("-cell", "--cell", type=float, default=None, help="Cell size")
    parser.add_argument("-mincell", "--min-cell", type=float, default=None, help="Cell size")
    parser.add_argument("-maxcell", "--max-cell", type=float, default=None, help="Cell size")
    parser.add_argument("-stepcell", "--step-cell", type=float, default=None, help="Cell size")
    parser.add_argument(
        "-pbc", "--pbc", type=bool, default=True, help="Periodic boundary conditions"
    )
    parser.add_argument(
        "-gpwmd", "--gpaw-mode", type=str, default="FD-TPSS", help="GPAW mode"
    )
    parser.add_argument(
        "-id",
        "--process-id",
        type=int,
        default=id_number,
        help="ID of the process (start time by default)",
    )

    args = parser.parse_args()

    filename = "params_convergence_{:s}_{:d}_{:d}".format(
        args.symbols, args.isomer_label, args.process_id
    )
    args.filename = filename

    return args, start_time

def cell_optimization(molecule, args, logger):
    
    if mpi.rank == 0:
        logger.info("CELL CONVERGENCE")
        logger.info(f'{"cell":<10}{"energy":<12}{"molecule":<12}{"atomization":<15}{"time":<18}')

    for a in np.arange(args.min_cell, args.max_cell, args.step_cell):
        #a = 0.2 * 4 * ng
        t1 = datetime.datetime.now()
        c = a / 2

        symbol = molecule.get_chemical_symbols()[0]
        atom = Atoms(symbol,
                     positions=[[c, c, c]],
                     cell=[a, a+0.001, a + 0.0002],
            )
        
        atom.set_pbc(args.pbc)
        params_calc_atom = get_calculator_params(name=args.gpaw_mode)
        params_calc_atom["txt"] = args.filename + ".out"


        calc_atom = GPAW(**params_calc_atom)
        atom.calc = calc_atom

        e1 = atom.get_potential_energy()
        
        molecule.set_cell([a, a, a])
        molecule.center()
        molecule.set_pbc(args.pbc)

        params_calc_mol = get_calculator_params(name=args.gpaw_mode)
        params_calc_mol["txt"] = args.filename + ".out"

        calc_mol = GPAW(**params_calc_mol)
        molecule.calc = calc_mol

        e2= molecule.get_potential_energy()
        t2 = datetime.datetime.now()

        if mpi.rank == 0:
            logger.info(f'{a:<10.3f}{e1:<12.3f}{e2:<12.3f}{(8 * e1 - e2):<15.3f}{(t2 - t1).total_seconds():<18.3f}')


def h_optimization(molecule, args, logger):
        
        if mpi.rank == 0:
            logger.info("H CONVERGENCE")
            logger.info(f'{"h":<8}{"gridpnts":<11}{"molecule":<14}{"atomization":<17}{"time":<20}')
            
        init_gridpoints = int(args.cell/0.24)
        final_gridpoints = int(args.cell/0.16)
        
        for ngrindpoints in np.arange(init_gridpoints, final_gridpoints, 4):
            t1 = datetime.datetime.now()
            
            h = args.cell / ngrindpoints

            symbol = molecule.get_chemical_symbols()[0]
            atom = Atoms(symbol,
                         positions=[[args.cell/2, args.cell/2, args.cell/2]],
                         cell=[args.cell, args.cell+0.001, args.cell + 0.0002],
                )

            atom.set_pbc(args.pbc)
            params_calc_atom = get_calculator_params(name=args.gpaw_mode)
            params_calc_atom["txt"] = args.filename + ".out"


            calc_atom = GPAW(**params_calc_atom)
            atom.calc = calc_atom

            e1 = atom.get_potential_energy()

            molecule.set_cell([args.cell, args.cell, args.cell])
            molecule.center()
            molecule.set_pbc(args.pbc)

            params_calc_mol = get_calculator_params(name=args.gpaw_mode)
            params_calc_mol["h"] = h
            params_calc_mol["txt"] = args.filename + ".out"

            calc_mol = GPAW(**params_calc_mol)
            molecule.calc = calc_mol

            e2 = molecule.get_potential_energy()
            t2 = datetime.datetime.now()

            if mpi.rank == 0:
                logger.info(f'{h:<8.3f}{ngrindpoints:<11}{e2:<14.3f}{8 * e1 - e2:<17.3f}{(t2 - t1).total_seconds():<20.3f}')

def main():

    args, start_time = parse_args()

    logger = init_logger()

    if mpi.rank == 0:

        logger.addHandler(logging.FileHandler(f"{args.filename}.log"))
        logger.info("PARAMETERS CONVERGENCE")

        for key, value in vars(args).items():
            if key != "filename" and key != "start_time":
                logger.info(f"{key}: {value}")

    molecule = get_molecule_isomer_minima(args.gpaw_mode,
        "{:s}".format(args.symbols), 
        "{:d}".format(args.isomer_label)
    )

    if args.cell is not None and args.min_cell is None and args.max_cell is None and args.step_cell is None:
        h_optimization(molecule, args, logger)
    
    if args.min_cell is not None and args.max_cell is not None and args.step_cell is not None and args.cell is None:
        cell_optimization(molecule, args, logger)

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

if __name__ == "__main__":
    main()

    

