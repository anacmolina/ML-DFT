import torch

from flonacomldft.utils.io_utils import (
    load_pickle_file
)

from flonacomldft.dft_calculator import DFTCalculator

from flonacomldft.internal_coordinates import (
    Structure,
    get_internal_coordinates_from_trajectory,
    get_construction_table,
    save_internal_coordinates_to_csv
)

mode_label = 1 # or 2

torch.manual_seed(42)

flow_model = load_pickle_file( "is{:d}_flow_dic_training.pkl".format(mode_label))['model']

N = 500

xs_samples = flow_model.sample(N)

flow_configurations = []

structure = Structure()
calculator = DFTCalculator()
calculator.initialize_calculator()

for i, x_sample in enumerate(xs_samples):
    molecule = structure.build_molecule(x_sample.detach())
    calculator.calculate_potential_energy(molecule, 
                            filename='ag6_flow_is{:d}_{:d}.out'.format(mode_label,i))

    flow_configurations.append(molecule)

xs_internal_coordinates = get_internal_coordinates_from_trajectory(flow_configurations, get_construction_table(), temperature=300)

save_internal_coordinates_to_csv(xs_internal_coordinates,
            get_construction_table(), 
            filename='is{:d}_flow_zmat.csv'.format(mode_label))
