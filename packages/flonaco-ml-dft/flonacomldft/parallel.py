import torch
import numpy as np
import gpaw.mpi as mpi

# set equal seed for all ranks for parallel computations

def set_seed(new_seed=None):

    ranks = np.arange(0, mpi.world.size)
    rank = mpi.world.rank
    comm = mpi.world.new_communicator(ranks)

    num_seed = np.array([0])

    if new_seed is None:
        if rank == 0:    
            num_seed = np.array([np.random.randint(1000000,9999999)])
        comm.broadcast(num_seed, 0)
    elif new_seed is not None:
        num_seed = np.array([new_seed])
        
    torch.manual_seed(num_seed[0])
    
    return int(num_seed[0])