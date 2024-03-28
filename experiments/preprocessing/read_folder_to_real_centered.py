# Script to read folders with GPAW outputs

import warnings
warnings.filterwarnings("ignore")
import argparse
from ase.io import read

import torch
import pandas as pd
from abflowmc.internal_coordinates import Coordinates_mapping
from abflowmc.internal_coordinates import save_internal_coordinates_to_csv
from abflowmc.utils.io_utils import save_ase_molecules_as_traj 
from abflowmc.internal_coordinates import load_DFTAdaptive_folder
from abflowmc.utils.io_utils import get_project_path

# define arguments argparser
parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('-path', '--path', type=str,)
parser.add_argument('-threads', '--threads', type=int, default=2)
parser.add_argument('-isomer', '--isomer-label', type=int,)
parser.add_argument('-etype', '--energy-type', type=str, default='dft')
parser.add_argument('-T', '--temperature', type=int, default=350)
parser.add_argument('-nruns', '--n-runs', type=int, default=30)
parser.add_argument('-nsteps', '--n-steps', type=int, default=20)
parser.add_argument('-nchains', '--n-chains', type=int, default=50)

args = parser.parse_args()

torch.set_num_threads(args.threads)

comps_id = int(args.path.split('/')[0].split('_')[-1])

coord_mapping = Coordinates_mapping(etype=args.energy_type)

zmats = load_DFTAdaptive_folder(args.path+'/DFTAdaptive', 
                                args.n_runs, 
                                args.n_steps, 
                                args.n_chains, 
                                args.isomer_label,
                                args.temperature,)

save_internal_coordinates_to_csv(zmats, 
                                construction_table=coord_mapping.construction_table, 
                                filename='is{:d}_proposals_zmat_{:d}.csv'.format(args.isomer_label, comps_id), 
                                add_cvs=True,
                                path=args.path)

print('ZMAT saved!')

xs_rc = coord_mapping.get_real_centered_from_internal(zmats[:, :12], 
                                                    isomer=args.isomer_label, 
                                                    temperature=args.temperature, 
                                                    energies=zmats[:, 12], 
                                                    logdetjacs=zmats[:, 14])

N = xs_rc[0].shape[0]

cols = ['rc_{:d}'.format(i) for i in range(12)] + ['potential_energy'] + ['isomer']

xs_rc = torch.cat((xs_rc[0], xs_rc[2].reshape(-1, 1), torch.ones((N, 1))*args.isomer_label), dim=1)

df = pd.DataFrame(xs_rc.detach().numpy(), columns=cols)

df.to_csv(args.path + '/is{:d}_proposals_xs_{:d}.csv'.format(args.isomer_label, comps_id), index=False)

print('RC saved!')