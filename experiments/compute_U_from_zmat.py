from ase.parallel import parprint as print

from datetime import datetime
import gpaw.mpi as mpi

import torch
import pandas as pd

from flonacomldft.dft_utils import Structure
from flonacomldft.data_utils import get_path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

cores = mpi.world.size

isomer = "is1"
i = 0

filename = get_path() + isomer + "_lcao_zmat.csv"
df = pd.read_csv(filename)
U = torch.Tensor(df.energies.to_numpy()).float()[i]
X = torch.Tensor(df.drop(["energies"], axis=1).to_numpy()).float()[i, :]

filename = "time_energy.out"
f = open(filename, "a")

ag6 = Structure()

startTime = datetime.now().timestamp()
ag6.calculate_potential_energy(X, txt="ag6_cores_" + str(cores) + ".out")
pot_energy = ag6.potential_energy
time_energy = datetime.now().timestamp() - startTime

diff = pot_energy - U

print(
    "Cores:{}".format(cores),
    "Time:{:.1f}s".format(time_energy),
    "U:{:0.3f}eV".format(pot_energy),
    "U md:{:0.3f}eV".format(U),
    "Diff:{:0.3f}eV".format(diff),
    end="  \n",
    file=f,
)

f.close()
