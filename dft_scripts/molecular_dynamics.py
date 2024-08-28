"""
Description: This script runs a molecular dynamics simulation.
"""

# import libraries
import time
import argparse
import logging

from gpaw import GPAW

from flonacomldft.utils.io_utils import set_str_date_to_int

#set up arguments
def parse_args():

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    id_number = set_str_date_to_int(start_time)

    parser = argparse.ArgumentParser(description="Run molecular dynamics simulation")

    parser.add_argument('-symbols', '--symbols', type=str, help='Symbols of the molecule')
    parser.add_argument('-isomer', '--isomer-label', type=str, help='Isomer label')
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

    args.start_time = start_time
    filename = "molecular_dynamics {:s}_{:s} {:d}".format(args.symbols, args.isomer_label, args.process_id)

    args.filename = filename

    return args