"""
Utils function to locate depending on host the files of previously saved
trajectories and trained models.
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch

from ase.io.trajectory import Trajectory

# Build a folder to save all trajectories
def trajectories_folder(name='trajectories', path=os.getcwd()):
    os.makedirs(path+'/'+name)

# Path to database folder (initial trajectories)
def get_path():
   if os.path.isdir('/mnt/home/amolina/ceph/database/'):
      ceph_home = '/mnt/home/amolina/ceph/database/'
   elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/database/'):
      ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/database/'
   elif os.path.isdir('/home/ana/ml_dft_project/database/'):
      ceph_home = '/home/ana/ml_dft_project/database/'
   elif os.path.isdir('/home/ana/assisting_sampling/database/'):
      ceph_home = '/home/ana/assisting_sampling/database/'
   elif os.path.isdir('/home/amolina/ml_dft_project/database/'):
      ceph_home = '/home/amolina/ml_dft_project/database/'
   else:
      raise RuntimeError('Data path not understood')
   return ceph_home

def get_project_path():
    return get_path().split('database')[0]

# def load_pickle_file(file):
def load_from_pickle(file):
    file_loaded = open(file, 'rb')
    _ = pickle.load(file_loaded)
    file_loaded.close()
    return _

# save_pickle_file(data, filename)
def save_pickle_file(data, filename):
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()

def load_zmat_csv(isomer):
    path = get_path()+isomer+'_lcao_zmat.csv'
    zmat = torch.tensor(pd.read_csv(path).to_numpy()).float()
    return zmat
    
def load_csv_file(isomer):
    path = get_path()+isomer+'_lcao_zmat.csv'
    df = pd.read_csv(path)
    U = torch.Tensor(df.energies.to_numpy()).float()
    X = torch.Tensor(df.drop(['energies'], axis=1).to_numpy()).float()
    return X, U 

# save_zmat_csv()
# TODO: write a function that saves the zmat 
