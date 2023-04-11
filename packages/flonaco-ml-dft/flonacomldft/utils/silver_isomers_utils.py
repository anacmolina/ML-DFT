import numpy as np
import pandas as pd
from ase import Atoms

isomers = {
    #'ag6_planar': {
    'is0': {
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[ 7.999201,  5.66275 ,  8.      ],
                               [ 7.986642, 10.350007,  8.      ],
                               [ 6.596777,  8.062411,  8.      ],
                               [ 5.320529,  5.699928,  8.      ],
                               [ 9.389729,  8.068932,  8.      ],
                               [10.678646,  5.717535,  8.      ]])},
    
    #'ag6_3d': {
    'is1': {
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[ 6.594017,  5.856863,  7.384655],
                               [ 9.405983,  5.856863,  7.384655],
                               [ 5.710476,  8.523162,  7.461228],
                               [ 8.      ,  7.764723,  8.6508  ],
                               [10.289524,  8.523162,  7.461228],
                               [ 8.      , 10.145127,  7.407456]])},
}

def get_molecule_isomer_minima(name, vacuum=None, **kwargs):
    
    if name in isomers:
        
        kwargs.update(isomers[name])
        molecule = Atoms(**kwargs)
    
    else:
        
        raise RuntimeError("Unknown isomer")
    
    if vacuum is not None:
    
        molecule.center(vacuum=vacuum)
    
    return molecule

# Construction table for both isomers, pandas dataframe (convention we chose)
def get_construction_table():

   index = np.append(0, np.append(np.arange(2,6), 1))
   construction_table = pd.DataFrame(index=index)
      
   construction_table['b'] = ['origin', 0, 2, 2, 4, 4]
   construction_table['a'] = ['e_z', 'e_z', 0, 3, 2, 2]
   construction_table['d'] = ['e_x', 'e_x', 'e_x', 0, 3, 3]
      
   return construction_table
