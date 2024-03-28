"""
Utils function to locate depending on host the files of previously saved
trajectories and trained models.
"""

import os
import pickle

import torch
import pandas as pd
from ase.io.trajectory import Trajectory



def get_project_path(path=None):
    """
    Get the path to the project folder.
    """
    
    if path is not None:
        return path
    else:
        raise Warning('Project path is not defined.')
        return "."

def get_path():
    """
    Get the path to the database folder.
    """
    return get_project_path() + '/database'


def load_pickle_file(filename, path=os.getcwd()):
    """
    Load a pickle file.
    Args:
        filename (str): the name of the file
        path (str): the path to the file
    Returns:
        object: the loaded object
    """
    
    file_loaded = open(path +'/' + filename, 'rb')
    _ = pickle.load(file_loaded)
    file_loaded.close()
    
    return _


def save_pickle_file(data, filename, path=os.getcwd()):
    """
    Save a pickle file.
    Args:
        data (object): the object to save
        filename (str): the name of the file
        path (str): the path to the file
    Returns:
        object: the loaded object   
    """

    outfile = open(path + '/' + filename, 'wb')
    pickle.dump(data, outfile)
    outfile.close()


def load_csv_file(filename, path=get_project_path(), dtype=torch.float32):
    """
    Load a csv file.
    Args:
        filename (str): the name of the file
        path (str): the path to the file
        dtype (torch.dtype): the data type
    Returns:
        torch.tensor: the loaded data
    """

    path =  path + '/' + filename
    zmat = torch.tensor(pd.read_csv(path).to_numpy()).to(dtype)
    
    return zmat


def save_csv_file(dataframe, filename, path=os.getcwd(), index=False):
    """
    Save a csv file.
    Args:
        dataframe (pd.DataFrame): the data to save
        filename (str): the name of the file
        path (str): the path to the file
        index (bool): whether to save the index
    """

    dataframe.to_csv(path + '/' + filename, index=index)


def save_ase_molecules_as_traj(configs, 
                               filename='configs.traj', 
                               path=os.getcwd()):
   """
   Save a list of ASE Atoms objects as a trajectory.
    Args:
         configs (list): the list of ASE Atoms objects
         filename (str): the name of the file
         path (str): the path to the file
   """

   traj = Trajectory(path + '/' + filename, 'w')
   for config in configs:
       traj.write(config)
   traj.close()


def save_json_args(args, script_name, id, path=os.getcwd()):
    """
    Save the arguments of a script to a json file.
    Args:
        args (Namespace): the arguments of the script
        script_name (str): the name of the script
        id (int): the id of the script
        path (str): the path to the file
    """

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


def set_str_date_to_int(date):
    """
    Set a string date to an integer.
    Args:
        date (str): the date
    Returns:
        int: the integer id
    """
    
    id = date.replace(' ', '').replace(':', '').replace('-', '')
    return int(id)

