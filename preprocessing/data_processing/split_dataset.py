import argparse

from flonacomldft.utils.io_utils import load_csv_file, save_json_args
from flonacomldft.utils.data_processing import split_data_from_dataframe
from flonacomldft.internal_coordinates import (
    get_construction_table,
    save_internal_coordinates_to_csv,
)

# Define arguments to parse from command line
parser = argparse.ArgumentParser(description="Prepare split dataset")
parser.add_argument("-ml", "--mode-label", type=int, default=0)
parser.add_argument("-o", "--origin", type=str, default="md")
parser.add_argument("-ts", "--train-size", type=float, default=0.8)
parser.add_argument("-ss", "--sk-seed", type=int, default=42)
parser.add_argument("-inpath", "--input-path", type=str, default="database/")
parser.add_argument("-outpath", "--output-path", type=str, default="database/")
parser.add_argument("-id", "--id", type=int, default=0)

args = parser.parse_args()

train_size = args.train_size
sk_seed = args.sk_seed

zmat = load_csv_file(
    args.input_path
    + "ag6_{:s}_zmat_is{:d}_{:d}.csv".format(args.origin, args.mode_label, args.id)
)

zmat_train_test = list(split_data_from_dataframe(zmat, train_size, sk_seed))

for zmat_, split_type in zip(zmat_train_test, ["train", "test"]):
    save_internal_coordinates_to_csv(
        zmat_,
        get_construction_table(),
        add_isomer=True,
        filename=args.output_path
        + "is{:d}_{:s}_{:s}_{:d}.csv".format(
            args.mode_label, args.origin, split_type, args.id
        ),
    )
    
args.algorithms = "split_dataset.py"

save_json_args(args, 'split_dataset')