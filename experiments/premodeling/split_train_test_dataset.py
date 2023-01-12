import torch

from flonacomldft.utils.io_utils import load_csv_file
from flonacomldft.utils.data_processing import split_data_from_dataframe
from flonacomldft.internal_coordinates import (
    get_construction_table,
    save_internal_coordinates_to_csv
)

train_size = 0.8
sk_seed = 42

mode_labels = [1, 2]

for mode_label, isomer_val in zip(mode_labels, [0, 1]):

    xs_md = load_csv_file("int_coords/is{:d}_lcao_zmat.csv".format(mode_label))
    xs_flow = load_csv_file("int_coords/is{:d}_flow_zmat.csv".format(mode_label))

    xs_md = torch.cat( (xs_md, torch.full((xs_md.shape[0], 1), isomer_val) ), dim=1 )
    xs_flow = torch.cat( (xs_flow, torch.full((xs_flow.shape[0], 1), isomer_val) ), dim=1 )

    x_md_train, x_md_test = split_data_from_dataframe(xs_md, train_size, sk_seed)
    x_flow_train, x_flow_test = split_data_from_dataframe(xs_flow, train_size, sk_seed)

    datasets = [[x_md_train, x_flow_train], [x_md_test, x_flow_test]]

    for xs, data_type in zip(datasets, ['train', 'test']):
        for xs_, data_origin in zip(xs, ['md','flow']):
            save_internal_coordinates_to_csv(xs_, 
                        get_construction_table(),
                        add_isomer=True, 
                        filename = 'datasets/is{:d}_{:s}_{:s}.csv'.format(mode_label, 
                                    data_origin,
                                    data_type) )