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
    """
    Get the path to the project folder.
    """
    if os.path.isdir('/mnt/home/amolina/ceph/adaptive-flow-mc'):
        ceph_home = '/mnt/home/amolina/ceph/adaptive-flow-mc'
    elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft'):
        ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft'
    elif os.path.isdir('/home/ana/adaptive-flow-mc'):
        ceph_home = '/home/ana/adaptive-flow-mc'
    elif os.path.isdir('/home/amolina/adaptive-flow-mc'):
        ceph_home = '/home/amolina/adaptive-flow-mc'    
    else:
        raise RuntimeError('Data path not understood')
    return ceph_home

# get database folder path
def get_path():
    """
    Get the path to the database folder.
    """
    return get_project_path() + '/database'

# TODO: delete this function
# created to save all the information of the simulations
def create_simulation_folder(name='flowMC', path=os.getcwd()):
    
    if os.path.isdir(path + '/' + name):
        pass
    else:
        os.makedirs(path + '/' + name)

# get simulation folder path
def get_simulation_folder_path(name, path=os.getcwd()):
    if os.path.isdir(path + '/' + name):
        return path.split(name)[0]
    else:
        raise RuntimeError('Folder not found')

# load pickle file
def load_pickle_file(filename, path=os.getcwd()):
    file_loaded = open(path +'/' + filename, 'rb')
    _ = pickle.load(file_loaded)
    file_loaded.close()
    return _

# save pickle file
def save_pickle_file(data, filename, path=os.getcwd()):
    outfile = open(path + '/' + filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()

# function that load csv data 
def load_csv_file(filename, path=get_project_path(), dtype=torch.float32):
    path =  path + '/' + filename
    zmat = torch.tensor(pd.read_csv(path).to_numpy()).to(dtype)
    return zmat

# function that saves data to csv
def save_csv_file(dataframe, filename, path=os.getcwd(), index=False):
    dataframe.to_csv(path + '/' + filename, index=index)


def save_ase_molecules_as_traj(configs, filename='configs.traj', path=os.getcwd()):
   traj = Trajectory(path + '/' + filename, 'w')
   for config in configs:
       traj.write(config)
   traj.close()

# add type of algorithm to args filename
def save_json_args(args, script_name, id, path=os.getcwd()):

    import json
    import numpy as np

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    filename_args = "args_{:s}_{:d}.json".format(script_name, id)

    with open(path + '/' + filename_args, "w") as outfile:
        json.dump(vars(args), outfile, cls=NpEncoder)

# change string date to int
def set_str_date_to_int(date):
    id = date.replace(' ', '').replace(':', '').replace('-', '')
    return int(id)

# change int date to string
def set_int_date_to_str(date_int):
    date_str = str(date_int)
    return date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8] + ' ' + date_str[8:10] + ':' + date_str[10:12] + ':' + date_str[12:14]

