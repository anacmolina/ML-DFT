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

mode_label = 0 # or 1

torch.manual_seed(42)

flow_model = load_pickle_file( "models/is{:d}_flow_dic_training.pkl".format(mode_label))['model']

N = 500

xs_samples = flow_model.sample(N)

flow_configurations = []

coord_maps = Coordinates_mapping()
calculator = DFTCalculator()
calculator.initialize_calculator()

for i, x_sample in enumerate(xs_samples):
    molecule = coord_maps.get_molecule_from_internal(x_sample.detach())
    calculator.calculate_potential_energy(molecule, 
                            filename='ag6_flow_is{:d}_{:d}.out'.format(mode_label,i))

    flow_configurations.append(molecule)

xs_internal_coordinates = coord_maps.get_internal_from_trajectory(flow_configurations, isomer=mode_label, temperature=300)

save_internal_coordinates_to_csv(xs_internal_coordinates,
            get_construction_table(), 
            filename='int_coords/is{:d}_flow_zmat.csv'.format(mode_label))
