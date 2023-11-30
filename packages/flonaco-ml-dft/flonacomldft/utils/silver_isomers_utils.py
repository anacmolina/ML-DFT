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
#}
#TODO: Fix this
#emt_isomers = {
    'emt_is0':{
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[ 7.53174253,  6.48514763,  5.        ],
                               [10.0723176 ,  5.6827374 ,  5.        ],
                               [ 5.30556767,  5.02138682,  5.        ],
                               [ 5.01737006,  7.60446966,  5.        ],
                               [ 7.15610969,  9.19211691,  5.        ],
                               [ 9.6463932 ,  8.24673131,  5.        ]])},

    'emt_is1':{
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[7.31189267, 5.21860471, 7.87202555],
                               [7.37558519, 5.01139179, 5.15837097],
                               [5.99252155, 7.5906818 , 7.65978813],
                               [5.00077846, 5.36783447, 6.44066432],
                               [8.36738115, 7.23469476, 6.3773041 ],
                               [6.05666685, 7.38362175, 4.94581578]])},
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


params_calc = {
    'LCAO': {'mode': 'lcao',
              'basis': 'pvalence.dz',
              'h': 0.2,
              'xc': 'PBE',
              'spinpol': True,
              'symmetry': 'off',
              #'nbands': -4,
              },

    'FD': {'mode': 'fd',
            'h': 0.18,
            'xc': 'PBE',
            'eigensolver': 'rmm-diis',
            'spinpol': True,
            'symmetry': 'off',
            #'nbands': -4,
            'parallel': dict(augment_grids=True,  # use all cores for XC/Poisson
                sl_auto=True,  # enable ScaLAPACK parallelization
                use_elpa=True),  # use ELPA for ScaLAPACK
            },
}

def get_molecule_calc_params(name='LCAO'):
    

    if name in params_calc:
    
        params = params_calc[name]

    else:
    
        raise RuntimeError("Undefined calculator parameters")

    
    return params

# Construction table for both isomers, pandas dataframe (convention we chose)
def get_construction_table():

   index = np.append(0, np.append(np.arange(2,6), 1))
   construction_table = pd.DataFrame(index=index)
      
   construction_table['b'] = ['origin', 0, 2, 2, 4, 4]
   construction_table['a'] = ['e_z', 'e_z', 0, 3, 2, 2]
   construction_table['d'] = ['e_x', 'e_x', 'e_x', 0, 3, 3]
      
   return construction_table