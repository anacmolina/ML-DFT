"""
Utils function to locate depending on host the files of previously saved
trajectories and trained models.
"""

import os
import pickle

import torch
import pandas as pd
from ase.io.trajectory import Trajectory


# path to database folder (initial trajectories)
def get_project_path():
    if os.path.isdir('/mnt/home/amolina/ceph/assist_sampling/'):
        ceph_home = '/mnt/home/amolina/ceph/assist_sampling/'
    elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'):
        ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'
    elif os.path.isdir('/home/ana/assisting_sampling/'):
        ceph_home = '/home/ana/assisting_sampling/'
    elif os.path.isdir('/home/amolina/assisting_sampling/'):
        ceph_home = '/home/amolina/assisting_sampling/'    
    else:
        raise RuntimeError('Data path not understood')
    return ceph_home

# get database folder path
def get_path():
    return get_project_path() + 'database/'

# created to save all the information of the simulations
def create_simulation_folder(name='flowMC', path=os.getcwd()):
    
    if os.path.isdir(path+'/'+name):
        pass
    else:
        os.makedirs(path+'/'+name)

# get simulation folder path
def get_simulation_folder_path(name, path=os.getcwd()):
    if os.path.isdir(path+'/'+name):
        return path.split(name)[0]
    else:
        raise RuntimeError('Folder not found')

# load pickle file
def load_pickle_file(filename, path=get_project_path()):
    file_loaded = open(path + filename, 'rb')
    _ = pickle.load(file_loaded)
    file_loaded.close()
    return _

# save pickle file
def save_pickle_file(data, filename, path=get_project_path()):
    outfile = open(path + filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()

# function that load csv data 
def load_csv_file(filename, path=get_project_path(), dtype=torch.float32):
    path =  path + filename
    zmat = torch.tensor(pd.read_csv(path).to_numpy()).to(dtype)
    return zmat

# function that saves data to csv
def save_csv_file(dataframe, filename, path=os.getcwd()):
    dataframe.to_csv(path + filename)


def save_ase_molecules_as_traj(configs, filename='configs.traj', path=os.getcwd()):
   traj = Trajectory(path + '/' + filename, 'w', atoms=configs)
   traj.close()

# add type of algorithm to args filename
def save_json_args(args, script_name, id, path=os.getcwd()):
    import json

    filename_args = "args_{:s}_{:d}.json".format(script_name, id)

    with open(path + '/' + filename_args, "w") as outfile:
        json.dump(vars(args), outfile)

def get_date_process_id():
    import time
    date = time.strftime('%H:%M:%S %d-%m-%Y')
    id = date.replace(' ', '').replace(':', '').replace('-', '')
    return date, int(id)