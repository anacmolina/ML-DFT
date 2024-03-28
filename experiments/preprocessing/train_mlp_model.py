import warnings
warnings.filterwarnings('ignore')

# standard library imports
import os
import time
import argparse

# scientific library imports
import torch
import numpy as np
import matplotlib.pyplot as plt

# flonaco library imports
# io handling
from abflowmc.utils.io_utils import (
    get_path,
    load_csv_file,
    load_pickle_file,
    save_csv_file,
    save_pickle_file,
    save_json_args,
    set_str_date_to_int
)

from abflowmc.utils.data_processing import split_data_from_dataframe

# coordinates handling
from abflowmc.internal_coordinates import Coordinates_mapping

# mlp training
from abflowmc.models.mlp import MLP
from abflowmc.train_mlp_from_data import train_mlp

# set seed and date_start
num_seed = np.random.randint(0, 100)
date_start = set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))

print("Random seed: {}".format(num_seed))
print("Date start: {}".format(date_start))

# set up parser
parser = argparse.ArgumentParser(description='Train MLP model from data.')
# execution parameters
parser.add_argument('-threads', '--threads', type=int, help='Set number of threads')
parser.add_argument('-pid', '--process-id', type=int, default=date_start, help='Set process id')
parser.add_argument('-rs', '--random_seed', type=int, default=num_seed, help='Set random seed')
parser.add_argument('-path', '--folder-path', type=str, default='andersen', help='Set path to data')
# system parameters
parser.add_argument('-isomer', '--isomer-label', type=int, default=0, help='Set isomer label')
parser.add_argument('-dataset', '--dataset', type=str, default='mlp', help='Set dataset')
# model parameters
parser.add_argument('-ni', '--n-iter', type=int, default=10, help='Set number of iterations')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Set learning rate')
parser.add_argument('-bs', '--batch-size', type=int, default=500, help='Set batch size')
parser.add_argument('-hdm', '--hidden-dim', type=int, default=64, help='Set hidden dimension')
parser.add_argument('-hdp', '--hidden-depth', type=int, default=3, help='Set hidden depth')
parser.add_argument('-us', '--use-scheduler', type=bool, default=False, help='Set scheduler')
parser.add_argument('-ss', '--step-scheduler', type=int, default=50, help='Set step size')
parser.add_argument('-Nmd', '--N-md-points', type=int, default=500, help='Set number of data points from MD')
parser.add_argument('-Nrd', '--N-random-points', type=int, default=500, help='Set number of data points from random gaussian sampling')

args = parser.parse_args()
args.date_start = date_start

dim = 12

# set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up number of threads for PyTorch
torch.set_num_threads(args.threads)
torch.manual_seed(args.random_seed)

print("Device: {}".format(device))
print("Threads: {}".format(torch.get_num_threads()))


# isomer labels
isomer_label = args.isomer_label

# path to the datasets
path_datasets = get_path() + '/' + args.folder_path + '/' + 'datasets'

# real center coordinates

# md dataset
flow_dataset = load_csv_file('is{:d}_{:s}_train.csv'.format(args.isomer_label, 
                                                            'flow'), path=path_datasets)[:args.N_md_points]

# random dataset
mlp_dataset = load_csv_file('is{:d}_{:s}_train.csv'.format(args.isomer_label, 
                                                            'mlp'), path=path_datasets)[:args.N_random_points]

# split data into train and test
xs_train_md, xs_test_md = list(split_data_from_dataframe(flow_dataset, 0.8, 42))
xs_train_mlp, xs_test_mlp = list(split_data_from_dataframe(mlp_dataset, 0.8, 42))

# concatenate datasets
xs_train = torch.cat((xs_train_md, xs_train_mlp) )
xs_test = torch.cat((xs_test_md, xs_test_mlp) )


print("Train data shape: {}".format(xs_train.shape))
print("Test data shape: {}".format(xs_test.shape))

# folder name
simulation_name = "is{:d}".format(isomer_label)


# path to save results
folder_to_save_results = 'results_mlp_{:s}_{:d}'.format(simulation_name, args.process_id)
path_to_save_results = os.getcwd() + '/' + folder_to_save_results
if not os.path.exists(path_to_save_results):
    os.makedirs(path_to_save_results)
    print('Folder created: ', path_to_save_results)

# mlp hyperparameters
n_hidden = args.hidden_dim
n_layers = args.hidden_depth
model = MLP([dim] +  [n_hidden] * n_layers + [1])

mlp_dic = train_mlp(model, 
                xs_train, 
                xs_test, 
                n_iter=args.n_iter,
                lr=args.learning_rate,
                bs=args.batch_size,
                use_scheduler=args.use_scheduler,
                step_scheduler=args.step_scheduler,
                with_tqdm=False,
                dim=dim)

# save end time
date_end = set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))
args.date_end = date_end

args.algorithm = 'train_mlp_model.py'

argparse_dic = vars(args)
mlp_dic['args'] = argparse_dic

# save output to pickle file
save_pickle_file(mlp_dic, "is{:d}_mlp_dic_training_{:d}.pkl".format(args.isomer_label, args.process_id), path=path_to_save_results)

# save arguments to json file
save_json_args(args, 'train_mlp_model', args.process_id, path=path_to_save_results)

# plot training and validation loss
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

axs[0].plot(mlp_dic['avg_train_losses'], label='train')
axs[0].plot(mlp_dic['avg_test_losses'], label='test')
axs[0].set_yscale('log')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')

axs[1].scatter(xs_train[:, dim].detach().numpy(), mlp_dic['model'](xs_train[:, :dim]).detach().numpy(), label='train')
axs[1].scatter(xs_test[:, dim].detach().numpy(), mlp_dic['model'](xs_test[:, :dim]).detach().numpy(), label='test')

axs[1].plot([xs_train[:, dim].min(), xs_train[:, dim].max()], [xs_train[:, dim].min(), xs_train[:, dim].max()], 'k--')

axs[1].set_xlabel('Energy')
axs[1].set_ylabel('Prediction')

axs[0].legend()
axs[1].legend()

plt.savefig(path_to_save_results + '/is{:d}_mlp_training_{:d}.png'.format(args.isomer_label, args.process_id))
