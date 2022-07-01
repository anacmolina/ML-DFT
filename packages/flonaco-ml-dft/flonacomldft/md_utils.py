#from ase.parallel import parprint as print

import os
import pickle

import numpy as np
import pandas as pd
import torch

import chemcoord as cc
from flonacomldft.dft_utils import get_construction_table

from ase import Atoms
from ase.io import read
from gpaw import GPAW
from ase.optimize import BFGS
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.io import Trajectory

# Get path to database
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

def load_from_pickle(file):
    file_loaded = open(file, 'rb')
    _ = pickle.load(file_loaded)
    file_loaded.close()
    return _

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

def shuffle_arr(vs, indexes):
    concat = lambda vs: torch.cat(vs)
    v = concat(vs)
    return v[indexes]

def rephase(zmat, angle=0, columns=['dihedral13']):
    for column in columns:
        phase = np.zeros(zmat[column].shape)
        phase[zmat[column]>angle] = -2*np.pi
        zmat[column] = zmat[column] + phase
    return zmat

def deg_to_rad(zmat):
    labels = zmat.columns.to_list()
    for label in labels[6:-1]:
        zmat[label] = np.deg2rad(zmat[label].tolist())
    return zmat

def get_internal_coordinates(traj):

    traj = traj
    construction_table = get_construction_table()
    energies = [traj_.get_potential_energy() for traj_ in traj]
    
    xyz = []
    for traj_ in traj:
        xyz.append(cc.Cartesian.from_ase_atoms(traj_))
    
    zmat = [xyz_.get_zmat(construction_table) for xyz_ in xyz]
    
    b = construction_table.b.to_numpy()
    a = construction_table.a.to_numpy()
    d = construction_table.d.to_numpy()
    ind = construction_table.index.to_numpy()

    label_b = ['bond'+str(i)+str(j) for i, j in zip(ind, b)]
    label_a = ['angle'+str(i)+str(j) for i, j in zip(ind, a)]
    label_d = ['dihedral'+str(i)+str(j) for i, j in zip(ind, d)]

    cols = label_b + label_a + label_d + ['energies']
    
    new_zmat = pd.DataFrame(columns = cols, index=np.arange(0, len(zmat), 1))
    
    for i in range(len(zmat)):
        new_zmat.iloc[i] = zmat[i].iloc[:, 2].tolist()+zmat[i].iloc[:, 4].tolist()+zmat[i].iloc[:, 6].tolist()+[energies[i]]
    
    new_zmat = deg_to_rad(new_zmat)
    new_zmat = rephase(new_zmat)

    new_zmat = new_zmat.drop(["bond0origin", "angle0e_z", "angle2e_z", "dihedral0e_x", "dihedral2e_x", "dihedral3e_x"], axis=1)

    new_zmat = new_zmat.to_numpy(dtype=np.float32)

    return torch.from_numpy(new_zmat).float()

# Molecular structures, minimums

def get_is1():
    pos = np.array([[7.9804600, 5.464791, 8.0],
                    [7.9611420, 10.16485, 8.0],
                    [6.5837500, 7.868667, 8.0],
                    [5.2993890, 5.513795, 8.0],
                    [9.3697860, 7.876240, 8.0],
                    [10.665305, 5.524828, 8.0]])
    isomer = Atoms('Ag6', positions=pos)
    return isomer

def get_is2():
    pos = np.array([[6.5910080, 5.595878, 7.139020],
                    [9.4089920, 5.595878, 7.139020],
                    [5.7157570, 8.272920, 7.161092],
                    [8.0000000, 7.529808, 8.358414],
                    [10.284243, 8.272920, 7.161092],
                    [8.0000000, 9.919833, 7.129457]])
    isomer = Atoms('Ag6', positions=pos)
    return isomer

# Running MD
def run_molecular_dynamics(molecule, iters, name):
    import gpaw.mpi as mpi
    rank = mpi.world.rank

    ceph_home = get_path()
    mol = molecule

    #Setting the cell

    mol.set_cell([16, 16, 16])
    mol.set_pbc(True)
    mol.center()

    # Building calculator
    calc = GPAW(mode="lcao", h=0.2, basis="pvalence.dz", spinpol=True, xc="PBE", symmetry="off", nbands = -4, txt='ag6_md_'+name+'.out')

    mol.set_calculator(calc)

    # Adding conditions to the MD simulation
    MaxwellBoltzmannDistribution(mol, temperature_K=300)
    Stationary(mol)
    ZeroRotation(mol)
    
    file = ceph_home+'ag6_'+name+'.traj'

    # Running the MD
    dyn = NVTBerendsen(mol, 5 * units.fs, taut = 50, temperature_K=300, trajectory=file)
    dyn.run(iters)
    
    # Getting the MD trajectory
    mpi.world.barrier()
    traj = Trajectory(file)

    return traj