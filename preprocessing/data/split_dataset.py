### Import modules
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

### Get start time and process id
date_start = time.strftime("%Y-%m-%d %H:%M:%S")
process_id = set_str_date_to_int(date_start)

### Define arguments to parse from command line
parser = argparse.ArgumentParser(description="Prepare split dataset")
parser.add_argument("-file", "--file", type=str)
parser.add_argument("-ml", "--mode-label", type=int, default=0)
parser.add_argument("-fm", "--for-model", type=str, default="flow")
parser.add_argument("-ts", "--train-size", type=float, default=0.8)
parser.add_argument("-ss", "--sk-seed", type=int, default=42)

args = parser.parse_args()

### Split parameters
train_size = args.train_size
sk_seed = args.sk_seed

### Load full dataset
zmat = load_csv_file(args.file, path = os.getcwd() + '/')[:5000]
zmat_train_test = list(split_data_from_dataframe(zmat, train_size, sk_seed))

for zmat_, split_type in zip(zmat_train_test, ["train", "test"]):
    save_internal_coordinates_to_csv(
        zmat_,
        get_construction_table(),
        add_isomer=True,
        filename="is{:d}_{:s}_{:s}.csv".format(
            args.mode_label, args.for_model, split_type
        ),
    )
    
args.algorithms = "split_dataset.py"
args.process_id = process_id

### Save params
save_json_args(args, 'split_dataset', process_id)