from ase.parallel import parprint as print
from datetime import datetime

print('Starting...\n')

print("Loading the libraries...")
import numpy as np
import torch
import time
import os
import sys
import pandas as pd

from flonacomldft.dft_utils import (
   get_path,
   Structure,
   AG6_construction_tables,
   Angles_transformation
)

print("Done\n")

ceph_home = get_path()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype=torch.float32

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)
print('Date:', date)

print('Device: %s...\n'%device)

print("Loading the database...")

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

symbols = np.full(6, 'Ag')
ct1 = AG6_construction_tables('is1')

ag6 = Structure(construction_table_=ct1, symbols_=symbols, Natoms_=len(symbols))

for i in range(5):
   startTime = datetime.now()
   ag6.calculate_potential_energy(np.array(X[i]))
   pot_energy = ag6.potential_energy
   print("Time: ", datetime.now() - startTime)

   print("PE calculate \t PE database \t Difference")
   print("----------------------------------------------------")
   print("%.5f \t %.5f \t %.5f"%(pot_energy, U[i], pot_energy-U[i]))

print("Done\n")

print("Finished")

