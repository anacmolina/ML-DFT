"""
Description: Minimal hopping script for DFT calculations
"""

# import libraries
import datetime
import argparse
import logging

from gpaw import GPAW
import gpaw.mpi as mpi
from ase.optimize.minimahopping import (
    MinimaHopping, 
    MHPlot
)

from flonacomldft.utils.io_utils import (
    set_str_date_to_int,
    init_logger
)
from flonacomldft.utils.silver_isomers_utils import (
    get_molecule_isomer_minima,
    get_calculator_params
)

#set up arguments
def parse_args():

    start_time = datetime.datetime.now()
    id_number = set_str_date_to_int(str(start_time).split('.')[0])

    parser = argparse.ArgumentParser(description="Run minimal hopping calculation")
    parser.add_argument('-symbols', '--symbols', type=str, help='Symbols of the molecule')
    parser.add_argument('-isomer', '--isomer-label', type=str, help='Isomer label')
    parser.add_argument('-cell', '--cell', type=float, help='Cell size')
    parser.add_argument('-pbc', '--pbc', type=bool, default=True, help='Periodic boundary conditions')
    parser.add_argument('-gpwmd', '--gpaw-mode', type=str, default='LCAO', help='GPAW mode')
    #Review this argument
    parser.add_argument('-etype', '--energy-type', type=str, default='dft', help='Energy type')
    parser.add_argument('-ediff0', '--ediff0', type=float, default=0.5, help='Energy convergence criterion')
    parser.add_argument('-T0', '--temperature0', type=float, default=1000, help='Initial temperature')
    parser.add_argument('-ns', '--n-steps', type=int, default=100, help='Number of steps')

    parser.add_argument('-id', '--process-id', type=int, default=id_number, help='ID of the process (start time by default)')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Seed for random number generator')

    args = parser.parse_args()

    filename = "minimal_hopping_{:s}_{:s}_{:d}".format(args.symbols, args.isomer_label, args.process_id)

    args.filename = filename

    return args, start_time

## set up logger
#def init_logger():
#
#    logging.basicConfig(
#        level=logging.INFO,
#        format="%(asctime)s - %(levelname)s - %(message)s",
#        datefmt='%Y-%m-%d %H:%M:%S'  # Time format
#    )
#
#    logger = logging.getLogger()
#
#    return logger

## set cell or vacuum
#def set_cell_or_vacuum(molecule, cell, vacuum):
#    
#    if cell is not None and vacuum is None:
#
#        molecule.set_cell([cell, cell, cell])
#        molecule.center()
#
#    elif cell is None and vacuum is not None:
#
#        molecule.center(vacuum=vacuum)
#
#    else:
#        
#        raise ValueError("Cell or vacuum cannot be set at the same time")
#
#    return molecule

def main():

    args, start_time = parse_args()
    filename = args.filename

    logger = init_logger()

    if mpi.rank == 0:

        logger.addHandler(logging.FileHandler(f"{args.filename}.log"))
        logger.info("MINIMAL HOPPING CALCULATION")

        for key, value in vars(args).items():
            if key != "filename" and key != "start_time":
                logger.info(f"{key}: {value}")

    #TODO: add logger
    molecule = get_molecule_isomer_minima(args.gpaw_mode,
        "{:s}".format(args.symbols), 
        "{:s}".format(args.isomer_label)
    )

    #molecule = set_cell_or_vacuum(molecule, 
    #                              args.cell, 
    #                              args.vacuum, 
    #                              )

    #fix for cell array
    molecule.set_cell([args.cell, args.cell, args.cell])

    molecule.set_pbc(args.pbc)

    params_calc = get_calculator_params(name=args.gpaw_mode)
    params_calc["txt"] = filename + ".out"

    calc = GPAW(**params_calc)

    molecule.calc = calc

    hop = MinimaHopping(molecule,
                    Ediff0=args.ediff0,
                    T0=args.temperature0)

    hop(totalsteps=args.n_steps)

    mhplot = MHPlot()
    mhplot.save_figure('summary.png')

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
