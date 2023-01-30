"""
Utils function to locate depending on host the files of previously saved
trajectories and trained models.
"""

import os
import pickle

import torch
import pandas as pd

# TODO: set database project path

# path to database folder (initial trajectories)
def get_path():
   if os.path.isdir('/mnt/home/amolina/ceph/database/'):
      ceph_home = '/mnt/home/amolina/ceph/database/'
   elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/database/'):
      ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/database/'
   elif os.path.isdir('/home/ana/assisting_sampling/database/'):
      ceph_home = '/home/ana/assisting_sampling/database/'
   else:
      raise RuntimeError('Data path not understood')
   return ceph_home

# get working folder path
def get_project_path():
    return get_path().split('database')[0]

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
def load_pickle_file(filename, path=get_path()):
    file_loaded = open(path + filename, 'rb')
    _ = pickle.load(file_loaded)
    file_loaded.close()
    return _

# save pickle file
def save_pickle_file(data, filename, path=get_path()):
    outfile = open(path + filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()

# function that load csv data 
def load_csv_file(filename, path=get_path(), dtype=torch.float32):
    path =  path + filename
    zmat = torch.tensor(pd.read_csv(path).to_numpy()).to(dtype)
    return zmat

# function that saves data to csv
def save_csv_file(dataframe, filename, path=os.getcwd()):
    dataframe.to_csv(path + filename)


 