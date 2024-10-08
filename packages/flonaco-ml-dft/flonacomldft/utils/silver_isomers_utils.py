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

    'FD-PBE':{
        
        'ag8': { '0': {
            'symbols': 'AgAgAgAgAgAgAgAg',
            'positions': np.array([[8.30220602, 5.88576036, 6.76685531],
                                    [9.33798813, 8.09293424, 5.45680286],
                                    [8.68441814, 6.05596154, 9.49246048],
                                    [5.63591088, 5.44850979, 6.15161202],
                                    [6.2220809 , 6.77007438, 8.49484034],
                                    [5.97166128, 9.49483705, 8.22049623],
                                    [8.50212845, 8.41334313, 8.06303507],
                                    [6.61236419, 8.03125162, 5.97053626]])
                                    },
         
                '1': {
            'symbols': 'AgAgAgAgAgAgAgAg',
            'positions': np.array([[5.42650701, 7.05129886, 6.53573959],
                                    [7.50601544, 5.59856154, 7.67432952],
                                    [7.96031563, 6.80801836, 5.2053203 ],
                                    [8.75008912, 6.6942953 , 9.83119628],
                                    [8.57171197, 9.46226277, 9.03044792],
                                    [9.56250968, 7.50971039, 7.30257465],
                                    [6.32541838, 7.79332421, 9.00409741],
                                    [7.30236679, 9.10139077, 6.66776416]])
                                    },
        }
            
    },

    'FD-TPSS':{
        
        'ag8': { '0': {
            'symbols': 'AgAgAgAgAgAgAgAg',
            'positions': np.array([[8.2842679 , 5.89086716, 6.77837364],
                                    [9.29417695, 8.0466582 , 5.44211908],
                                    [8.67376269, 6.06830772, 9.47876245],
                                    [5.64385462, 5.45691239, 6.15886458],
                                    [6.23753012, 6.77683411, 8.47989391],
                                    [5.98133214, 9.48047407, 8.21004935],
                                    [8.48628215, 8.39195561, 8.03673882],
                                    [6.61109658, 8.01635147, 5.98505657]])
                                    },
         
                '1': {
            'symbols': 'AgAgAgAgAgAgAgAg',
            'positions': np.array([[5.45098599, 7.06594317, 6.53813709],
                                    [7.51104358, 5.6276032 , 7.67203766],
                                    [7.97062715, 6.84486785, 5.2385692 ],
                                    [8.7456157 , 6.69971576, 9.82391301],
                                    [8.57767624, 9.4515579 , 9.03362843],
                                    [9.54807344, 7.51261819, 7.32278677],
                                    [6.36240927, 7.80240507, 8.98056724],
                                    [7.31580546, 9.09301214, 6.68917464]])
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
            'h': 0.18,
            'xc': 'PBE',
            #'eigensolver': 'rmm-diis',
            #'spinpol': True,
            #'symmetry': 'off',
            #'nbands': -4, #TODO: ScaLAPACK parallelization has problems on maestria PC
            #'parallel': dict(augment_grids=True,  # use all cores for XC/Poisson
            #    sl_auto=True,  # enable ScaLAPACK parallelization
            #    use_elpa=True # use ELPA for ScaLAPACK
            #    ),  
            },

    'FD-TPSS': {'mode': 'fd',
            'h': 0.18,
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