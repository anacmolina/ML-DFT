import numpy as np
from ase import Atoms

isomers = {
    'Ag6_planar': {
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[7.9804600, 5.464791, 8.0],
                        [7.9611420, 10.16485, 8.0],
                        [6.5837500, 7.868667, 8.0],
                        [5.2993890, 5.513795, 8.0],
                        [9.3697860, 7.876240, 8.0],
                        [10.665305, 5.524828, 8.0]])},
    
    'Ag6_3d': {
        'symbols': 'AgAgAgAgAgAg',
        'positions': np.array([[6.5910080, 5.595878, 7.139020],
                        [9.4089920, 5.595878, 7.139020],
                        [5.7157570, 8.272920, 7.161092],
                        [8.0000000, 7.529808, 8.358414],
                        [10.284243, 8.272920, 7.161092],
                        [8.0000000, 9.919833, 7.129457]])}
            }

def Ag6Isomers(name, vacuum=None, **kwargs):
    
    if name in isomers:
        
        kwargs.update(isomers[name])
        molecule = Atoms(**kwargs)
    
    else:
        
        raise RuntimeError("Unknown isomer")
    
    if vacuum is not None:
    
        molecule.center(vacuum=vacuum)
    
    return molecule