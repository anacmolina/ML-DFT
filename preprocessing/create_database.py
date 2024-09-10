"""
Description: This script creates a database from molecular dynamics trajectories
"""

# import libraries
import os
import datetime
import argparse
import logging

import torch

from ase.io.trajectory import Trajectory

from flonacomldft.utils.io_utils import set_str_date_to_int
from flonacomldft.internal_coordinates import (
    Coordinates_mapping,
    internal_coordinates_to_csv
)
from flonacomldft.utils.silver_isomers_utils import get_construction_table
from flonacomldft.utils.data_processing import split_data_from_dataframe


# set up arguments
def parse_args():

    start_time = datetime.datetime.now()
    id_number = set_str_date_to_int(str(start_time).split('.')[0])

    parser = argparse.ArgumentParser(description="Database creation")
    parser.add_argument("-symbols", "--symbols", type=str, help="Symbols of the molecule")
    parser.add_argument("-isomer", "--isomer-label", type=int, help="Isomer label")
    parser.add_argument("-etype", "--energy-type", type=str, help="Energy type")
    parser.add_argument("-gpwmd", "--gpaw-mode", type=str, default="FD-TPSS", help="GPAW mode")
    parser.add_argument("-trajfile", "--trajectory-file", type=str, help="Trajectory file")
    parser.add_argument("-T", "--temperature", type=int, default=350, help="Temperature")
    parser.add_argument("-N", "--num-samples", type=int, default=None, help="Number of samples")
    parser.add_argument("-low", "--low-index", type=int, default=None, help="Low index")
    parser.add_argument("-ts", "--train-size", type=float, default=0.8)
    parser.add_argument("-ss", "--sk-seed", type=int, default=42)
    parser.add_argument(
        "-id",
        "--process-id",
        type=int,
        default=id_number,
        help="ID of the process (start time by default)",
    )

    args = parser.parse_args()

    filename = "database_{:s}_{:d}_{:d}".format(
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

# main function

def main():

    # parse arguments
    args, start_time = parse_args()

    # set up logger
    logger = init_logger()
    logger.addHandler(logging.FileHandler(f"{args.filename}.log"))
    logger.info("DATABASE CREATION")

    traj = Trajectory(args.trajectory_file)    

    logger.info("set up low index and number of samples")
    if args.low_index is None:
        args.low_index = 0

    if args.num_samples is None:
        args.num_samples = len(traj)

    for key, value in vars(args).items():
            if key != "filename" and key != "start_time":
                logger.info(f"{key}: {value}")

    logger.info("defining construction table")
    construction_table = get_construction_table(chemical_formula=args.symbols)
    
    coord_maps = Coordinates_mapping(construction_table=construction_table,
                                     chemical_formula=args.symbols,
                                     calculator_label=args.gpaw_mode)

    logger.info("calculating internal coordinates")
    zmats = coord_maps.get_internal_from_trajectory(traj[args.low_index:args.low_index+args.num_samples], 
                                                       isomer=args.isomer_label, 
                                                       temperature=args.temperature, 
                                                       max_samples=args.num_samples).detach()
    zmats_df = internal_coordinates_to_csv(zmats, 
                            construction_table=construction_table, 
                            filename=args.filename + '.csv', 
                            path=os.getcwd())

    N_atoms = coord_maps.Natoms

    logger.info("calculating real centered coordinates")
    coord_rc, logdetjacs_rc, energies_rc = coord_maps.get_real_centered_from_internal(zmats[:, :(N_atoms*3-6)], 
                                                      isomer=args.isomer_label,
                                                      temperature=args.temperature, 
                                                      energies=zmats[:, (N_atoms*3-6)],
                                                      logdetjacs=zmats[:, (N_atoms*3-6)+2])

    rc = torch.cat([coord_rc, 
                energies_rc.reshape(-1, 1),
                torch.ones(coord_rc.shape[0]).reshape(-1, 1)*args.isomer_label,
                logdetjacs_rc.reshape(-1, 1),
                zmats[:, -2:]], dim=1)

    rc_df = internal_coordinates_to_csv(rc, 
                            construction_table=construction_table,
                            filename=args.filename + '_rc.csv', 
                            path=os.getcwd())
    
    logger.info("splitting data into train and test")
    train, test = (split_data_from_dataframe(rc_df, args.train_size, args.sk_seed))

    zmats_df.to_csv(args.filename + '_zmat.csv', index=False)
    rc_df.to_csv(args.filename + '_rc.csv', index=False)
    train.to_csv(args.filename + '_train.csv', index=False)
    test.to_csv(args.filename + '_test.csv', index=False)

    end_time = datetime.datetime.now()

    logger.info("database creation started at {:s}".format(str(start_time).split('.')[0]))
    logger.info("database creation finished at {:s}".format(str(end_time).split('.')[0]))

    logger.info("database took: {:f} seconds".format(
         (end_time-start_time).total_seconds())
        )

if __name__ == "__main__":
    main()

