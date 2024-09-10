"""
Description: This script is used to optimize the structure of a molecule
"""

# import libraries
import datetime
import argparse
import logging

from gpaw import GPAW, mpi
from ase.optimize import BFGS

from flonacomldft.utils.io_utils import set_str_date_to_int
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
    parser.add_argument("-cell", "--cell", type=float, help="Cell size")
    parser.add_argument("-vacuum", "--vacuum", type=float, help="Vacuum size")
    parser.add_argument(
        "-pbc", "--pbc", type=bool, default=True, help="Periodic boundary conditions"
    )
    parser.add_argument(
        "-gpwmd", "--gpaw-mode", type=str, default="FD-TPSS", help="GPAW mode"
    )
    parser.add_argument(
        "-fmax", "--fmax", type=float, default=0.01, help="Maximum force"
    )
    parser.add_argument(
        "-id",
        "--process-id",
        type=int,
        default=id_number,
        help="ID of the process (start time by default)",
    )

    args = parser.parse_args()

    filename = "optimization_{:s}_{:d}_{:d}".format(
        args.symbols, args.isomer_label, args.process_id
    )
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
# TODO: Generalize this function to accept arrays
def set_cell_or_vacuum(molecule, cell, vacuum):
    
    if cell is not None and vacuum is None:

        molecule.set_cell([cell, cell, cell])
        molecule.center()

    elif cell is None and vacuum is not None:

        molecule.center(vacuum=vacuum)

    else:
        
        raise ValueError("Cell or vacuum cannot be set at the same time")

    return molecule


def main():

    args, start_time = parse_args()
    filename = args.filename

    logger = init_logger()

    if mpi.rank == 0:

        logger.addHandler(logging.FileHandler(f"{args.filename}.log"))
        logger.info("STRUCTURE OPTIMIZATION")

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

    opt = BFGS(molecule, logfile=filename + '.log', trajectory=filename + ".traj")
    opt.run(args.fmax)

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
