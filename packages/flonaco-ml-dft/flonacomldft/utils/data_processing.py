# TODO: Move funtion to another file, dataprocessing folder in experiments

import torch


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

