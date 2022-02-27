print('Starting...\n')

print("Loading the libraries...")
import numpy as np
import torch
import time
import os
import sys
import pandas as pd

from flonacomldft.dft_utils import (
    Structure,
    AG6_construction_tables,
    Angles_tranformation
    )

print("Done\n")

if os.path.isdir('/mnt/home/amolina/ceph/database'):
   ceph_home = '/mnt/home/amolina/ceph/database'
elif os.path.isdir('/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'):
   ceph_home = '/Users/marylou/Dropbox/Prof/Experiments/_ceph/ml-dft/'
elif os.path.isdir('/home/anacristina/ml_dft_project/database/'):
    ceph_home = '/home/anacristina/ml_dft_project/database/'
elif os.path.isdir('/home/amolina/ml_dft_project/database/'):
    ceph_home = '/home/amolina/ml_dft_project/database/'
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

print(len(sys.argv))

if(len(sys.argv)>1):

   isomer = sys.argv[1]
   mode = sys.argv[2]

else:
   isomer = "is1"
   mode = "lcao"

name = isomer+"_"+mode
df = pd.read_csv(ceph_home + name +"_zmat.csv")

print("Done\n")

print("Setting the input...")

U = df.energies.to_numpy()
X = df.drop(['energies'], axis=1).to_numpy()

print("Done\n")

print("Calculating the energy for one configuration")

i=0
symbols = np.full(6, 'Ag')
ct1 = AG6_construction_tables('is1')
ag6 = Structure(construction_table_=ct1, symbols_=symbols, Natoms_=len(symbols))

ag6.calculate_potential_energy(np.array(X[i]))
pot_energy = ag6.potential_energy

print("PE calculate \t PE database \t Difference")
print("----------------------------------------------------")
print("%.5f \t %.5f \t %.5f"%(pot_energy, U[i], pot_energy-U[i]))

os.remove("ag6.out")
print("Done\n")

print("Finished")

