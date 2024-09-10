# libraries
import os
import time
import argparse

from flonacomldft.utils.io_utils import load_csv_file, save_json_args, get_path
from flonacomldft.utils.data_processing import split_data_from_dataframe
from flonacomldft.internal_coordinates import (
    save_internal_coordinates_to_csv,
)
from flonacomldft.utils.io_utils import set_str_date_to_int

# get date and set process id
date = set_str_date_to_int(time.strftime("%Y-%m-%d %H:%M:%S"))

# define arguments to parse from command line
parser = argparse.ArgumentParser(description="Prepare split dataset")
parser.add_argument("-file", "--file", type=str)
parser.add_argument("-isomer", "--isomer-label", type=int,)
parser.add_argument("-fm", "--for-model", type=str, default="flow")
parser.add_argument("-ts", "--train-size", type=float, default=0.8)
parser.add_argument("-ss", "--sk-seed", type=int, default=42)
parser.add_argument("-N", "--num-samples", type=int, default=5000)
parser.add_argument("-low", "--low-index", type=int, default=0)
parser.add_argument("-pid", "--process-id", type=int, default=date)

args = parser.parse_args()
 
dim = 12

# dataset params
train_size = args.train_size
sk_seed = args.sk_seed

# full dataset
xs = load_csv_file(args.file, path = os.getcwd() + '/')[args.low_index : args.low_index + args.num_samples]
xs_train_test = list(split_data_from_dataframe(xs, train_size, sk_seed))

folder = args.file.split("/")[-1].split(".")[0].split("_")[2] + "/datasets"
columns = ['rc{:d}'.format(i) for i in range(dim)]

# save train and test dataset in separate files
for xs_, split_type in zip(xs_train_test, ["train", "test"]):
    save_internal_coordinates_to_csv(
        xs_,
        columns=columns,
        filename="is{:d}_{:s}_{:s}.csv".format(
            args.isomer_label, args.for_model, split_type
        ),
        path=get_path() + '/' + folder,
    )
    
args.algorithms = "split_dataset.py"
args.date = date

# save params
save_json_args(args, 'split_dataset', args.process_id)