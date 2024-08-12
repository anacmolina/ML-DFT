import numpy as np
import pandas as pd
from ase import Atoms

isomers = {

    'ag6': {'0': {
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[ 7.999201,  5.66275 ,  8.      ],
                               [ 7.986642, 10.350007,  8.      ],
                               [ 6.596777,  8.062411,  8.      ],
                               [ 5.320529,  5.699928,  8.      ],
                               [ 9.389729,  8.068932,  8.      ],
                               [10.678646,  5.717535,  8.      ]])},
    
    
            '1': {
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[ 6.594017,  5.856863,  7.384655],
                               [ 9.405983,  5.856863,  7.384655],
                               [ 5.710476,  8.523162,  7.461228],
                               [ 8.      ,  7.764723,  8.6508  ],
                               [10.289524,  8.523162,  7.461228],
                               [ 8.      , 10.145127,  7.407456]])},
    },

    'ag8': { '0': {
        'symbols': 'AgAgAgAgAgAgAgAg',
        'positions': np.array([[ 5.65151856,  7.77493647,  6.81677953],
                               [ 7.77019127,  6.32370671,  7.89527028],
                               [ 8.17812089,  7.59266523,  5.44603277],
                               [ 9.01514428,  7.42842472, 10.05241499],
                               [ 8.81561256, 10.19395979,  9.28444985],
                               [ 9.81102768,  8.28157329,  7.51867661],
                               [ 6.57749245,  8.53215712,  9.26724566],
                               [ 7.50507551,  9.85670492,  6.91851273]])},

            '1': {
        'symbols': 'AgAgAgAgAgAgAgAg',
        'positions': np.array([[7.68747184, 5.42559624, 6.2787369 ],
                               [8.70408048, 7.66638587, 5.03402521],
                               [8.05321644, 5.54541435, 9.00974873],
                               [5.03144193, 5.03443986, 5.63797309],
                               [5.57785271, 6.27955338, 8.03771243],
                               [5.35611683, 9.01942241, 7.8149322 ],
                               [7.89192442, 7.94539875, 7.65520354],
                               [5.98700311, 7.62000358, 5.52181059]])},
    },

#TODO: Include the following isomers in the next isomers update
#emt_isomers = {
#    'emt_is0':{
#        'symbols': 'AgAgAgAgAgAg',
#        'positions': np.array([[ 7.53174253,  6.48514763,  5.        ],
#                               [10.0723176 ,  5.6827374 ,  5.        ],
#                               [ 5.30556767,  5.02138682,  5.        ],
#                               [ 5.01737006,  7.60446966,  5.        ],
#                               [ 7.15610969,  9.19211691,  5.        ],
#                               [ 9.6463932 ,  8.24673131,  5.        ]])},
#
#    'emt_is1':{
#        'symbols': 'AgAgAgAgAgAg',
#        'positions': np.array([[7.31189267, 5.21860471, 7.87202555],
#                               [7.37558519, 5.01139179, 5.15837097],
#                               [5.99252155, 7.5906818 , 7.65978813],
#                               [5.00077846, 5.36783447, 6.44066432],
#                               [8.36738115, 7.23469476, 6.3773041 ],
#                               [6.05666685, 7.38362175, 4.94581578]])},

} 


def get_molecule_isomer_minima(symbols, isomer_label, vacuum=5.0, **kwargs):

    symbols = symbols.lower()
    isomer_label = str(isomer_label)    

    if symbols in isomers:

        if isomer_label in isomers[symbols]:

            kwargs.update(isomers[symbols][isomer_label])
            molecule = Atoms(**kwargs)

            if vacuum is not None:
    
                molecule.center(vacuum=vacuum)
    
        else:
    
            raise RuntimeError("Unknown isomer")

    else:
    
        raise RuntimeError("Unrecognized symbols")
    
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