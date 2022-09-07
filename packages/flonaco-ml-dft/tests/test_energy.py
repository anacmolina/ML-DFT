from ase.parallel import parprint as print
from datetime import datetime

print('Starting...\n')

print("Loading the libraries...")
import numpy as np
import torch
import time
import pandas as pd

from flonacomldft.dft_utils import (
   Structure
)

from flonacomldft.data_utils import get_path

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

df_is1 = pd.read_csv(ceph_home + "is1_lcao_zmat.csv")
df_is2 = pd.read_csv(ceph_home + "is2_lcao_zmat.csv")

print("Done\n")

print("Setting the input...")

U_is1 = df_is1.energies.to_numpy()
X_is1 = df_is1.drop(['energies'], axis=1).to_numpy()

U_is2 = df_is2.energies.to_numpy()
X_is2 = df_is2.drop(['energies'], axis=1).to_numpy()

print("Done\n")

print("Calculating the energy for the two isomers\n")

ag6 = Structure()

print("t [s] \t PE [eV] \t PE_md [eV] \t Diff [eV]")
print("------------------------------------------------")
   
i = 0

for x, u in zip([X_is1[i], X_is2[i]], [U_is1[i], U_is2[i]] ):
   startTime = datetime.now()
   ag6.calculate_potential_energy(np.array(x))
   pot_energy = ag6.potential_energy
   t = (datetime.now() - startTime).total_seconds()

   print("%.2f \t %.3f \t %.3f \t %.3f"%(t, pot_energy, u, pot_energy-u))

print("Done\n")

print("Finished")

