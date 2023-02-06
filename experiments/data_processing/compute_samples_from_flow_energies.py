#TODO: Run in the cluster
import torch

from flonacomldft.utils.io_utils import (
    load_pickle_file
)

from flonacomldft.dft_calculator import DFTCalculator

from flonacomldft.internal_coordinates import (
    Coordinates_mapping,
    get_construction_table,
    save_internal_coordinates_to_csv
)

torch.manual_seed(42)

mode_label = 0 # or 1
N = 3 #2500

flow_model = load_pickle_file( "models/is{:d}_flow_dic_training.pkl".format(mode_label))['model']
coord_mapping = Coordinates_mapping()

xs_samples = flow_model.sample(N)
zs_samples, logdetjacs = coord_mapping.get_internal_from_real_centered(xs_samples, isomer=mode_label)

calculator = DFTCalculator()
calculator.initialize_calculator()

flow_configurations = []
 
for i, zmat_sample in enumerate(zs_samples):
    molecule = coord_mapping.build_molecule_from_zmat(zmat_sample.detach())
    calculator.calculate_potential_energy(molecule, 
                            filename='ag6_flow_is{:d}_{:d}.out'.format(mode_label,i))

    flow_configurations.append(molecule)

internal_coordinates = coord_mapping.get_internal_from_trajectory(flow_configurations, isomer=mode_label, temperature=300)

# TODO: save configs as a trajectory
save_internal_coordinates_to_csv(internal_coordinates,
            get_construction_table(), 
            filename='int_coords/is{:d}_flow_zmat.csv'.format(mode_label))
