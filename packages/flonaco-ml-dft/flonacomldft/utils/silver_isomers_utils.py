import numpy as np
import pandas as pd
from ase import Atoms

isomers = {
    #'ag6_planar': {
    'dft_is0': {
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[ 7.999201,  5.66275 ,  8.      ],
                               [ 7.986642, 10.350007,  8.      ],
                               [ 6.596777,  8.062411,  8.      ],
                               [ 5.320529,  5.699928,  8.      ],
                               [ 9.389729,  8.068932,  8.      ],
                               [10.678646,  5.717535,  8.      ]])},
    
    #'ag6_3d': {
    'dft_is1': {
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[ 6.594017,  5.856863,  7.384655],
                               [ 9.405983,  5.856863,  7.384655],
                               [ 5.710476,  8.523162,  7.461228],
                               [ 8.      ,  7.764723,  8.6508  ],
                               [10.289524,  8.523162,  7.461228],
                               [ 8.      , 10.145127,  7.407456]])},
#}
#TODO: Fix this
#emt_isomers = {
    'is0':{
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[ 7.53174253,  6.48514763,  5.        ],
                               [10.0723176 ,  5.6827374 ,  5.        ],
                               [ 5.30556767,  5.02138682,  5.        ],
                               [ 5.01737006,  7.60446966,  5.        ],
                               [ 7.15610969,  9.19211691,  5.        ],
                               [ 9.6463932 ,  8.24673131,  5.        ]])},

    'is1':{
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[7.31189267, 5.21860471, 7.87202555],
                               [7.37558519, 5.01139179, 5.15837097],
                               [5.99252155, 7.5906818 , 7.65978813],
                               [5.00077846, 5.36783447, 6.44066432],
                               [8.36738115, 7.23469476, 6.3773041 ],
                               [6.05666685, 7.38362175, 4.94581578]])},
} 


def get_molecule_isomer_minima(name, etype='dft', vacuum=None, **kwargs):
    
    dict_isomers = None

    if etype == 'dft':
        dict_isomers = isomers
    elif etype == 'emt':
        dict_isomers = emt_isomers
    else:
        raise RuntimeError("Unknown potential type")

    if name in dict_isomers:
    
        kwargs.update(dict_isomers[name])
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
