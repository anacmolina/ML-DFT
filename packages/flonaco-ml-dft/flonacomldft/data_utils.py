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
   elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'):
      ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'
   elif os.path.isdir('/home/ana/ml_dft_project/database/'):
      ceph_home = '/home/ana/ml_dft_project/database/'
   elif os.path.isdir('/home/amolina/ml_dft_project/database/'):
      ceph_home = '/home/amolina/ml_dft_project/database/'
   else:
      raise RuntimeError('Data path not understood')
   return ceph_home

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
"""
def load_is_csv(isomer):
    ceph_home = get_path()
    file = '_lcao_zmat.csv'
    u_init = torch.tensor(pd.read_csv(ceph_home + isomer + file)['energies'].to_numpy()).float()
    x_init = torch.tensor(pd.read_csv(ceph_home + isomer + file).drop('energies', axis=1).to_numpy()).float()
    if isomer=='is1':
        count_init = torch.zeros(x_init.shape[0])
    elif isomer=='is2':
        count_init = torch.ones(x_init.shape[0])
    else:
        raise RuntimeError('Can not find isomer!')
    return x_init, u_init, count_init
"""
# save_zmat_csv()

