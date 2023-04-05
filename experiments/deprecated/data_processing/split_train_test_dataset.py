import torch

from flonacomldft.utils.io_utils import load_csv_file
from flonacomldft.utils.data_processing import split_data_from_dataframe
from flonacomldft.internal_coordinates import (
    get_construction_table,
    save_internal_coordinates_to_csv
)

train_size = 0.8
sk_seed = 42

mode_labels = [0, 1]

for mode_label in mode_labels:

    zmats_md = load_csv_file("int_coords/is{:d}_md_zmat.csv".format(mode_label))
    zmats_flow = load_csv_file("int_coords/is{:d}_flow_zmat.csv".format(mode_label))

    zmats_md_train, zmats_md_test = split_data_from_dataframe(zmats_md, train_size, sk_seed)
    zmats_flow_train, zmats_flow_test = split_data_from_dataframe(zmats_flow, train_size, sk_seed)

    datasets = [[zmats_md_train, zmats_flow_train], [zmats_md_test, zmats_flow_test]]

    for xs, data_type in zip(datasets, ['train', 'test']):
        for xs_, data_origin in zip(xs, ['md','flow']):
            save_internal_coordinates_to_csv(xs_, 
                        get_construction_table(),
                        add_isomer=True, 
                        filename = 'datasets/is{:d}_{:s}_{:s}.csv'.format(mode_label, 
                                    data_origin,
                                    data_type) )