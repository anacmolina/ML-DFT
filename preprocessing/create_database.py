"""
Description: This script creates a database from molecular dynamics trajectories
"""
# import libraries
import datetime
import argparse
import logging


from flonacomldft.utils.io_utils import set_str_date_to_int
from flonacomldft.internal_coordinates import Coordinates_mapping

# set up arguments
def parse_args():

    start_time = datetime.datetime.now()
    id_number = set_str_date_to_int(str(start_time).split('.')[0])

    parser = argparse.ArgumentParser(description="Database creation")
    parser.add_argument(
        "-file", "--file", type=str, help="Path to the trajectory file"
        )
    parser.add_argument("-symbols", "--symbols", type=str, help="Symbols of the molecule")
    parser.add_argument("-isomer", "--isomer-label", type=int, help="Isomer label")
    parser.add_argument("-etype", "--energy-type", type=str, help="Energy type")
    parser.add_argument("-gpwmd", "--gpaw-mode", type=str, default="FD-TPSS", help="GPAW mode")
    parser.add_argument("-T", "--temperature", type=int, default=350, help="Temperature")
    parser.add_argument("-N", "--num-samples", type=int, default=None, help="Number of samples")
    parser.add_argument("-low", "--low-index", type=int, default=0, help="Low index")
    parser.add_argument(
        "-id",
        "--process-id",
        type=int,
        default=id_number,
        help="ID of the process (start time by default)",
    )

    args = parser.parse_args()

    args = parser.parse_args()

    filename = "optimization_{:s}_{:s}_{:d}".format(
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

    logger.info("DATABASE CREATION")
    for key, value in vars(args).items():
            if key != "filename" and key != "start_time":
                logger.info(f"{key}: {value}")

    coord_maps = Coordinates_mapping()

