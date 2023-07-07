# libraries
import os
import time
import argparse

from flonacomldft.utils.io_utils import load_csv_file, save_json_args
from flonacomldft.utils.data_processing import split_data_from_dataframe
from flonacomldft.internal_coordinates import (
    get_construction_table,
    save_internal_coordinates_to_csv,
)
from flonacomldft.utils.io_utils import set_str_date_to_int

# get date and set process id
date = set_str_date_to_int(time.strftime("%Y-%m-%d %H:%M:%S"))

# define arguments to parse from command line
parser = argparse.ArgumentParser(description="Prepare split dataset")
parser.add_argument("-file", "--file", type=str)
parser.add_argument("-isomer", "--isomer-label", type=int, default=0)
parser.add_argument("-fm", "--for-model", type=str, default="flow")
parser.add_argument("-ts", "--train-size", type=float, default=0.8)
parser.add_argument("-ss", "--sk-seed", type=int, default=42)
parser.add_argument("-N", "--num-samples", type=int, default=5000)
parser.add_argument("-pid", "--process-id", type=int, default=date)

args = parser.parse_args()

# dataset params
train_size = args.train_size
sk_seed = args.sk_seed

# full dataset
zmat = load_csv_file(args.file, path = os.getcwd() + '/')[:args.num_samples]
zmat_train_test = list(split_data_from_dataframe(zmat, train_size, sk_seed))

# save train and test dataset in separate files
for zmat_, split_type in zip(zmat_train_test, ["train", "test"]):
    save_internal_coordinates_to_csv(
        zmat_,
        get_construction_table(),
        add_isomer=True,
        filename="is{:d}_{:s}_{:s}.csv".format(
            args.isomer_label, args.for_model, split_type
        ),
    )
    
args.algorithms = "split_dataset.py"
args.date = date

# save params
save_json_args(args, 'split_dataset', args.process_id)