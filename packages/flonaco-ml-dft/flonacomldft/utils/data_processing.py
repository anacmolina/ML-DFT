# TODO: Move funtion to another file, dataprocessing folder in experiments

import torch
from flonacomldft.internal_coordinates import Coordinates_mapping
from flonacomldft.utils.io_utils import load_csv_file, get_path

# split data

def split_data_from_dataframe(dataset, train_size, sk_seed):
    
    from torch.utils.data import random_split

    length = int(dataset.shape[0]*train_size)
    lengths = [length, dataset.shape[0]-length]

    split = random_split(dataset=dataset,
                        lengths=lengths,
                        generator= torch.Generator().manual_seed(sk_seed))

    dataset_splitted = [dataset[data.indices] for data in split]

    return tuple(dataset_splitted)

#TODO: Change flow by a parameter in the function
def load_datasets(md, isomer_id, name, real_centered=True):
    
    zmats = {data_type: load_csv_file('is{:d}_{:s}_{:s}.csv'.format(isomer_id, name, data_type),
                                      get_path() + '/{:s}/datasets'.format(md))
             for data_type in ['train', 'test']}


    if real_centered:
        if 'emt' in md:
            etype = 'emt'
        else:
            etype = 'dft'
    
        coord_mapping = Coordinates_mapping(etype=etype)
    
        xs = {data_type: coord_mapping.get_real_centered_from_internal(
                                    zmats[data_type][:, :12],
                                    zmats[data_type][:, 14],
                                    isomer=isomer_id,
                                    energies=zmats[data_type][:, 12]
                                    ) for data_type in ['train', 'test'] }
        
        xs = {data_type: torch.cat((xs[data_type][0], xs[data_type][2].reshape(-1, 1), 
                                 zmats[data_type][:, 13].reshape(-1, 1), 
                                 ), dim=1) for data_type in ['train', 'test']}    
    return xs

