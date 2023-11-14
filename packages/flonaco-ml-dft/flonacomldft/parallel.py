#TODO: DELETE THIS FILE! DEPRECATED!
### Import modules
import torch
import numpy as np

def set_seed(new_seed=None):
    """Set the seed for the random number generator."""

    num_seed = np.array([0])

    ### If no seed is given, generate a random seed
    if new_seed is None:

        ### Parallet setup of random seed
        import gpaw.mpi as mpi

        ranks = np.arange(0, mpi.world.size)
        rank = mpi.world.rank
        comm = mpi.world.new_communicator(ranks)

        ### Share the seed with all ranks
        if rank == 0:    
            num_seed = np.array([np.random.randint(0,100)])
        comm.broadcast(num_seed, 0)

    ### If a seed is given, use it    
    elif new_seed is not None:
        num_seed = np.array([new_seed])

    ### Set the seed                   
    torch.manual_seed(num_seed[0])
    
    return int(num_seed[0])