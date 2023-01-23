# TODO: Delete this file

import matplotlib.pyplot as plt
import pandas as pd

from ase.visualize.plot import plot_atoms

from flonacomldft.internal_coordinates import Coordinates_mapping
from flonacomldft.FES.plotter2 import Plotter
from flonacomldft.utils.data_utils import get_path
from flonacomldft.collective_variables import get_CVs

def plot_sample(x):
   fig, ax = plt.subplots()
   ag6 = Coordinates_mapping()
   plot_atoms(ag6.get_molecule_from_internal(x), ax)

def plotting_fes_db(train_data=None):
      
   plotting = Plotter(400, 'Ag6')
   plotting.readfile(get_path() + 'unrotated_300.txt')
   
   fig, ax = plotting.plot_fes(0.1, 300, delta2=1, shift=1.5)
   
   if train_data is not None:

      db1 = pd.read_csv(get_path() + 'is1_lcao_zmat.csv')
      db1 = db1.drop(['energies'], axis=1)
      db1 = db1.to_numpy()
   
      c_db1, r_db1 = get_CVs(db1)

      db2 = pd.read_csv(get_path() + 'is2_lcao_zmat.csv')
      db2 = db2.drop(['energies'], axis=1)
      db2 = db2.to_numpy()
   
      c_db1, r_db1 = get_CVs(db1)
      c_db2, r_db2 = get_CVs(db2)
   
      ax.plot(c_db1, r_db1, 'c.', label='DB is1')
      ax.plot(c_db2, r_db2, 'c.', label='DB is2')
   
   return fig, ax