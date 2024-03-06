"""
IO functions for reading and writing data
"""

import os

import torch
import pandas as pd
from ase.io.trajectory import Trajectory


DATABASE_PATH = None
PROJECT_PATH = Noneb



def set_project_path(path):
    """
    Set the path to the project folder.
    Args:
        path (str): path to the project folder
    """

    if os.path.isdir(path):
        PROJECT_PATH = path
    else:
        raise RuntimeError('Project path not understood!')
    

def get_project_path():
    """
    Get the path to the project folder.
    """

    if PROJECT_PATH is None:

        if os.path.isdir('/mnt/home/amolina/ceph/adaptive-flow-mc'):
            PROJECT_PATH = '/mnt/home/amolina/ceph/adaptive-flow-mc'
        elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft'):
            PROJECT_PATH = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft'
        elif os.path.isdir('/home/ana/adaptive-flow-mc'):
            PROJECT_PATH = '/home/ana/adaptive-flow-mc'
        elif os.path.isdir('/home/amolina/adaptive-flow-mc'):
            PROJECT_PATH = '/home/amolina/adaptive-flow-mc'    
        else:
            raise RuntimeError('Project path not understood!')
        
    return PROJECT_PATH


def set_database_path(path):
    """
    Set the path to the database folder.
    Args:
        path (str): path to the database folder
    """

    if os.path.isdir(path):
        DATABASE_PATH = path
    else:
        raise RuntimeError('Database path not understood!')
    