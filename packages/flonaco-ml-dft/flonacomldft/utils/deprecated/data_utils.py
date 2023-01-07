# TODO: change file name to io_utils.py

"""
Utils function to locate depending on host the files of previously saved
trajectories and trained models.
"""

import os
import pickle

import torch
import pandas as pd

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

# build to save all the information of the simulations
def set_working_folder(name='flowMC', path=os.getcwd()):
    os.makedirs(path+'/'+name)

# TODO: get working folder path
#def get_working_folder_path()

# build a folder to save all information
#def trajectories_folder(name='trajectories', path=os.getcwd()):
#    os.makedirs(path+'/'+name)



# TODO: set database folder path
# TODO: get database folder path

def get_project_path():
    return get_path().split('database')[0]

# load pickle file
def load_pickle_file(file):
    file_loaded = open(file, 'rb')
    _ = pickle.load(file_loaded)
    file_loaded.close()
    return _

# save pickle file
def save_pickle_file(data, filename):
    outfile = open(filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()

# function that load csv data 
def load_csv_file(filename, path=get_path()):
    path =  path + filename
    zmat = torch.tensor(pd.read_csv(path).to_numpy()).float()
    return zmat

# function that saves data to csv
def save_csv_file(dataframe, filename, path=os.getcwd()):
    dataframe.to_csv(path + filename)


#def load_csv_file(isomer):
#    path = get_path()+isomer+'_lcao_zmat.csv'
#    df = pd.read_csv(path)
#    U = torch.Tensor(df.energies.to_numpy()).float()
#    X = torch.Tensor(df.drop(['energies'], axis=1).to_numpy()).float()
#    return X, U 
 

def split_data_from_dataframe(df, train_size, sk_seed):
    
    import sklearn.model_selection

    zmat = torch.tensor(df.to_numpy()).float()
    x = zmat[:, :-1].numpy()
    y = zmat[:, -1].numpy()

    arrays = [x, y]

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(*arrays,
                                                      test_size=None,
                                                      train_size=train_size,
                                                      random_state=sk_seed,
                                                      shuffle=True,
                                                      stratify=None)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    return x_train, x_test, y_train, y_test 