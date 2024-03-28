import warnings
warnings.filterwarnings('ignore')

from abflowmc.observables.optical_spectra import compute_optical_spectra
import os
import torch
import numpy as np
from ase.io import read

import argparse
from abflowmc.internal_coordinates import Coordinates_mapping
from abflowmc.utils.io_utils import load_pickle_file, get_project_path
import gpaw.mpi as mpi

ranks = np.arange(0, mpi.world.size)
rank = mpi.rank

parser = argparse.ArgumentParser(description='Compute optical spectra for a molecule.')
parser.add_argument('-isomer', '--isomer', type=int, help='isomer to compute optical spectra for')
parser.add_argument('-path', '--path', type=str, default=os.getcwd(), help='path to write results to')
parser.add_argument('-aid', '--adaptive-id', type=int, help='adaptive sampling id')
parser.add_argument('-range', '--range', type=int, nargs='+', default=0, help='range of chains to compute spectra for')
parser.add_argument('-etype', '--energy-type', type=str, default='dft', help='Energy type used in the adaptive')
parser.add_argument('-row', '--row', type=int, default=-1, help='Number of rows in the plot')

args = parser.parse_args()

#molecule = get_molecule_isomer_minima('is{:d}'.format(args.isomer))

adaptive_path = {'dft': '1-adaptive/results_adaptive_is'+str(args.isomer)+'_'+str(args.adaptive_id)+'/adaptive_sampling_is'+str(args.isomer)+'_'+str(args.adaptive_id)+'.pkl',
                    'mlp': '2-adaptive-mlp/results_adaptive_is'+str(args.isomer)+'_'+str(args.adaptive_id)+'/adaptive_sampling_is'+str(args.isomer)+'_'+str(args.adaptive_id)+'.pkl'}

xs = torch.cat(load_pickle_file(adaptive_path[args.energy_type], path=get_project_path())['xs'])[args.row]

coord_mapping = Coordinates_mapping()

for i in range(args.range[0], args.range[1]):
    print('Computing Sample {:d}'.format(i))
    x = xs[i].reshape(1, -1)
    molecule = coord_mapping.build_molecule_from_real_centered(x, isomer=args.isomer)[0]
    molecule.center(vacuum=8)

    if rank == 0:

        if not os.path.exists(args.path+'/sample_'+str(i)):
            os.makedirs(args.path+'/sample_'+str(i))

        from ase.visualize.plot import plot_atoms
        import matplotlib.pyplot as plt

        plt.figure()
        plot_atoms(molecule)
        plt.savefig(args.path+'/sample_'+str(i)+'/molecule.png')

    mpi.world.barrier()

    try:
        compute_optical_spectra(molecule, args.path+'/sample_'+str(i))
        print('Success!')
    except:
        print('Failed!')
