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
from flonacomldft.utils.io_utils import (
    get_path,
    load_csv_file,
    load_pickle_file,
    save_csv_file,
    save_pickle_file,
    save_json_args,
    set_str_date_to_int
)

# coordinates handling
from flonacomldft.internal_coordinates import Coordinates_mapping

# mlp training
from flonacomldft.models.mlp import MLP
from flonacomldft.train_mlp_from_data import train_mlp

# set seed and date_start
num_seed = np.random.randint(0, 100)
date_start = set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))

print("Random seed: {}".format(num_seed))
print("Date start: {}".format(date_start))

# set up parser
parser = argparse.ArgumentParser(description='Train MLP model from data.')
# execution parameters
parser.add_argument('-threads', '--threads', type=int, default=None, help='Set number of threads')
parser.add_argument('-pid', '--process-id', type=int, default=date_start, help='Set process id')
parser.add_argument('-rs', '--random_seed', type=int, default=num_seed, help='Set random seed')
parser.add_argument('-path', '--folder-path', type=str, default='andersen/', help='Set path to data')
# system parameters
parser.add_argument('-isomer', '--isomer-label', type=int, default=0, help='Set isomer label')
parser.add_argument('-etype', '--energy-type', type=str, default='dft', help='Set energy type')
parser.add_argument('-T', '--temperature', type=float, default=350, help='Set temperature')
# model parameters
parser.add_argument('-ni', '--n-iter', type=int, default=10, help='Set number of iterations')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Set learning rate')
parser.add_argument('-bs', '--batch-size', type=int, default=500, help='Set batch size')
parser.add_argument('-hdm', '--hidden-dim', type=int, default=64, help='Set hidden dimension')
parser.add_argument('-hdp', '--hidden-depth', type=int, default=3, help='Set hidden depth')
parser.add_argument('-us', '--use-scheduler', type=bool, default=False, help='Set scheduler')
parser.add_argument('-ss', '--step-schedule', type=int, default=50, help='Set step size')

args = parser.parse_args()
args.date_start = date_start

# set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up number of threads for PyTorch
if args.threads is not None:
    torch.set_num_threads(args.threads)

print("Device: {}".format(device))
print("Threads: {}".format(torch.get_num_threads()))

# set random seed
torch.manual_seed(args.random_seed)

# isomer labels
isomer_label = args.isomer_label
energy_type = args.energy_type

# path to the datasets
path_datasets = get_path() + '/' + args.folder_path + '/' + 'datasets'

# real center coordinates
coord_mapping = Coordinates_mapping(etype=energy_type)

# TODO: is this the best way to get the dataset?
# TODO: real center data in a file already, add collective varibles to csv file 
def get_dataset(name, path, isomer_label):
    
    zmats = load_csv_file("is{:d}_{:s}.csv".format(isomer_label, name), path)
    xs = coord_mapping.get_real_centered_from_internal(
                                    zmats[:, :12],
                                    zmats[:, 14],
                                    isomer=isomer_label,
                                    energies=zmats[:, 12],
                                    temperature=args.temperature,
                                    )

    xs = torch.cat((xs[0], xs[2].reshape(-1, 1), 
                                 #zmats[:, 13].reshape(-1, 1), 
                                 #x[1].reshape(-1, 1),
                                 ), dim=1)
    
    #xs = xs.flatten(start_dim=0, end_dim=1).to(torch.float32)

    return xs, zmats

xs_train_mlp, zmat_train_mlp = get_dataset('mlp_train', path_datasets, isomer_label)
xs_test_mlp, zmat_test_mlp = get_dataset('mlp_test', path_datasets, isomer_label)

print('mlp shape: ', xs_train_mlp.shape, xs_test_mlp.shape)

xs_train_md, zmat_train_md = get_dataset('flow_train', path_datasets, isomer_label)
xs_test_md, zmat_test_md = get_dataset('flow_test', path_datasets, isomer_label)

print('md: ', xs_train_md.shape, xs_test_md.shape)

print(xs_train_mlp.shape, xs_train_md.shape)

xs_train = torch.cat((xs_train_md, xs_train_mlp) )
xs_test = torch.cat((xs_test_md, xs_test_mlp) )

print("Train data shape: {}".format(xs_train.shape))
print("Test data shape: {}".format(xs_test.shape))

simulation_name = "is{:d}".format(isomer_label)

# path to save results
folder_to_save_results = 'results_mlp_{:s}_{:d}'.format(simulation_name, args.process_id)
path_to_save_results = os.getcwd() + '/' + folder_to_save_results
if not os.path.exists(path_to_save_results):
    os.makedirs(path_to_save_results)
    print('Folder created: ', path_to_save_results)

### MLP hyperparameters
n_hidden = args.hidden_dim
n_layers = args.hidden_depth
model = MLP([12] +  [n_hidden] * n_layers + [1])

mlp_dic = train_mlp(model, 
                xs_train, 
                xs_test, 
                n_iter=args.n_iter,
                lr=args.learning_rate,
                use_scheduler=args.use_scheduler,
                step_schedule=args.step_schedule,
                with_tqdm=False)

# save end time
date_end = set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))
args.date_end = date_end

args.algorithm = 'train_mlp_model'

argparse_dic = vars(args)
mlp_dic['args'] = argparse_dic

# save output to pickle file
save_pickle_file(mlp_dic, "is{:d}_mlp_dic_training_{:d}.pkl".format(args.isomer_label, args.process_id), path=path_to_save_results)

# save arguments to json file
save_json_args(args, 'train_mlp_model', args.process_id, path=path_to_save_results)

# plot training and validation loss
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

axs[0].plot(np.log(mlp_dic['train_loss']), label='train')
axs[0].plot(np.log(mlp_dic['test_loss']), label='test')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('log loss')

axs[1].scatter(xs_train[:, 12].detach().numpy(), mlp_dic['model'](xs_train[:, :12]).detach().numpy(), label='train')
axs[1].scatter(xs_test[:, 12].detach().numpy(), mlp_dic['model'](xs_test[:, :12]).detach().numpy(), label='test')

axs[1].set_xlabel('Energy')
axs[1].set_ylabel('Prediction')

axs[0].legend()
axs[1].legend()

plt.savefig(path_to_save_results + '/is{:d}_mlp_training_{:d}.png'.format(args.isomer_label, args.process_id))
