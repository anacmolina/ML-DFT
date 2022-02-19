print('Starting...\n')

print("Loading the libraries...")
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import pandas as pd

from flonacomldft.dft_utils import (
    Structure, 
    AG6_construction_tables,
    Angles_tranformation
    )

print("Done\n")

if os.path.isdir('/mnt/home/amolina/ceph/'):
   ceph_home = '/mnt/home/amolina/ceph/'
elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'):
   ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'
elif os.path.isdir('/home/anacristina/Monte Carlo Project/Trajectories/Data_4/'):
    ceph_home = '/home/anacristina/Monte Carlo Project/Trajectories/Data_4/'
else:
   raise RuntimeError('Data path not understood')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype=torch.float32

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)
print('Date:', date)

print('Device: %s...\n'%device)

print("Loading the database...")

data_home = ceph_home + 'Database/'

df = pd.read_csv(data_home + 'is1_zmat.csv')
df = df.drop(['Unnamed: 0'], axis=1)

print("Done\n")

print("Setting the input...")

U = df.energies.to_numpy()
X = df.drop(['energies'], axis=1).to_numpy()

#x_tensor = torch.from_numpy(X).float()
#U_tensor = torch.from_numpy(U).float()

print("Done\n")

print("Calculating the energy for one configuration")

i=0
symbols = np.full(6, 'Ag')
ct1 = AG6_construction_tables('is1')
ag6 = Structure(construction_table_=ct1, symbols_=symbols, Natoms_=len(symbols))

ag6.calculate_potential_energy(X[i])
pot_energy = ag6.potential_energy

print(pot_energy, U[i], pot_energy-U[i])
os.remove("ag6.out")
print("Done\n")

print("Finished")
