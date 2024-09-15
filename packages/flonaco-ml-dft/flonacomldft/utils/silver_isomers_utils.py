import numpy as np
import pandas as pd
from ase import Atoms

isomers = {

    'LCAO':{

        'ag6': {'0': {
            'symbols': 'AgAgAgAgAgAg',
            'positions': np.array([[ 7.999201,  5.66275 ,  8.      ],
                                   [ 7.986642, 10.350007,  8.      ],
                                   [ 6.596777,  8.062411,  8.      ],
                                   [ 5.320529,  5.699928,  8.      ],
                                   [ 9.389729,  8.068932,  8.      ],
                                   [10.678646,  5.717535,  8.      ]])
                                   },


                '1': {
            'symbols': 'AgAgAgAgAgAg',
            'positions': np.array([[ 6.594017,  5.856863,  7.384655],
                                   [ 9.405983,  5.856863,  7.384655],
                                   [ 5.710476,  8.523162,  7.461228],
                                   [ 8.      ,  7.764723,  8.6508  ],
                                   [10.289524,  8.523162,  7.461228],
                                   [ 8.      , 10.145127,  7.407456]])
                                   },
        },
    },

    'FD-TPSS':{
        
        'ag8': { '0': {
            'symbols': 'AgAgAgAgAgAgAgAg',
            'positions': np.array([[7.64540586, 5.42790676, 6.25450181],
                                   [8.66670582, 7.63903168, 4.9925928 ],
                                   [7.98645493, 5.51387775, 8.9744954 ],
                                   [5.02250016, 4.98742349, 5.58867918],
                                   [5.56026287, 6.2567737 , 7.94796841],
                                   [5.31922645, 8.97604913, 7.73840742],
                                   [7.82181303, 7.87704663, 7.58740301],
                                   [5.97609856, 7.55149242, 5.49325651]])
                                    },
         
                '1': {
            'symbols': 'AgAgAgAgAgAgAgAg',
            'positions': np.array([[5.04220011, 6.4580867 , 6.35989439],
                                   [7.14285719, 5.03203647, 7.43419719],
                                   [7.55573577, 6.26285282, 5.00532727],
                                   [8.37927369, 6.08711868, 9.58855927],
                                   [8.1875086 , 8.84955357, 8.82901712],
                                   [9.1411193 , 6.94917615, 7.08259654],
                                   [5.98923417, 7.19636439, 8.78110993],
                                   [6.89166867, 8.50043396, 6.48108234]])
                                    },
        }
            
    },

    'EMT':{

        'ag6':{ '0': {
            'symbols': 'AgAgAgAgAgAg',
            'positions': np.array([[ 7.53174253,  6.48514763,  5.        ],
                                   [10.0723176 ,  5.6827374 ,  5.        ],
                                   [ 5.30556767,  5.02138682,  5.        ],
                                   [ 5.01737006,  7.60446966,  5.        ],
                                   [ 7.15610969,  9.19211691,  5.        ],
                                   [ 9.6463932 ,  8.24673131,  5.        ]])
                                   },

                '1':{
            'symbols': 'AgAgAgAgAgAg',
            'positions': np.array([[7.31189267, 5.21860471, 7.87202555],
                                   [7.37558519, 5.01139179, 5.15837097],
                                   [5.99252155, 7.5906818 , 7.65978813],
                                   [5.00077846, 5.36783447, 6.44066432],
                                   [8.36738115, 7.23469476, 6.3773041 ],
                                   [6.05666685, 7.38362175, 4.94581578]])
                                   },
        },

    },

} 

def get_molecule_isomer_minima(calculator_label, symbols, isomer_label, vacuum=None, **kwargs):

    calculator_label = calculator_label.upper()
    symbols = symbols.lower()
    isomer_label = str(isomer_label)

    if calculator_label in isomers:    

        if symbols in isomers[calculator_label]:    

            if isomer_label in isomers[calculator_label][symbols]:

                kwargs.update(isomers[calculator_label][symbols][isomer_label])
                molecule = Atoms(**kwargs)

                if vacuum is not None:
                
                    molecule.center(vacuum=vacuum)

            else:
            
                raise RuntimeError("Unknown isomer")

        else:
            
            raise RuntimeError("Unrecognized symbols")
        
    else:

        raise RuntimeError("Unknown calculator label")
    
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

    'FD-PBE': {'mode': 'fd',
            'h': 0.2,
            'xc': 'PBE',
            #'eigensolver': 'rmm-diis',
            #'spinpol': True,
            'symmetry': 'off',
            #'nbands': -4, #TODO: ScaLAPACK parallelization has problems on maestria PC
            'parallel': dict(augment_grids=True,  # use all cores for XC/Poisson
                sl_auto=True,  # enable ScaLAPACK parallelization
                use_elpa=True # use ELPA for ScaLAPACK
                ),  
            },

    'FD-TPSS': {'mode': 'fd',
            'h': 0.2,
            'xc': 'TPSS',
            #'eigensolver': 'rmm-diis',
            #'spinpol': True,
            #'symmetry': 'off',
            #'nbands': -4, #TODO: ScaLAPACK parallelization has problems on maestria PC
            #'parallel': dict(augment_grids=True,  # use all cores for XC/Poisson
            #    sl_auto=True,  # enable ScaLAPACK parallelization
            #    use_elpa=True # use ELPA for ScaLAPACK
            #    ),  
            },
}

def get_calculator_params(name='LCAO'):
    
    if name in params_calc:
    
        params = params_calc[name]

    else:
    
        raise RuntimeError("Undefined calculator parameters")

    
    return params

# Construction table for both isomers, pandas dataframe (convention we chose)
def get_construction_table(chemical_formula='ag6'):
     
    chemical_formula = chemical_formula.lower()
   
    if chemical_formula == 'ag6':
       index = np.append(0, np.append(np.arange(2,6), 1))
       construction_table = pd.DataFrame(index=index)
       
       construction_table['b'] = ['origin', 0, 2, 2, 4, 4]
       construction_table['a'] = ['e_z', 'e_z', 0, 3, 2, 2]
       construction_table['d'] = ['e_x', 'e_x', 'e_x', 0, 3, 3]
       
    elif chemical_formula == 'ag8':

       index = np.array([6, 0, 1, 2, 4, 5, 7, 3])
       construction_table = pd.DataFrame(index=index)  

       construction_table['b'] = ['origin', 6, 0, 0, 2, 4, 5, 7]
       construction_table['a'] = ['e_z', 'e_z', 6, 1, 0, 2, 4, 5]
       construction_table['d'] = ['e_x', 'e_x', 'e_x', 6, 1, 0, 2, 4]
       
    else:
       
         raise RuntimeError("Undefined construction table")
      
    return construction_table