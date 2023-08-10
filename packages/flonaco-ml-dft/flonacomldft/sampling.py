"""
Script with all sampling methods. 

"""

import time
import numpy as np
import torch
import tqdm

#from ase.parallel import parprint as print
from ase.units import kB

#kb = 8.617333262e-5

def run_metropolis(
    model,
    init,
    n_chains,
    n_steps,
    dim=12,
    id_run=None,
    energy_type=None,
    frac_dft=0.2,
    mlp_models=None,
    mixture=False,
    T=300,
    with_tqdm=False,
    return_ratio = False,
    return_proposals = False,
    dft_folder_name = None,
    scheduler = 5,
    update_weigth = False,
    alpha=0.1,
):
    """
    Run Metropolis-Hastings algorithm to sample from a model.
    
    Args:
        model (torch.nn.Module): model to sample from
        init (torch.Tensor): initial positions of shape (n_chains, n_dim + 2) - xs, us, isomers
        n_chains (int): number of chains to run
        n_steps (int): number of steps to run
        id_run (int): id of the run
        energy_type (str): type of energy to use (dft, mlp, dft+mlp)
        frac_dft (float): fraction of chains for which to use DFT per step
        mlp_models (torch.nn.Module): MLP model to use for energy calculation
        mixture (bool): whether the model is a mixture
        T (float): temperature in K
        with_tqdm (bool): whether to use tqdm progress bar
    
    Returns:
        dictionary reporting all the progress
    """

    # add data dimension and data structure compatibility

    assert init.shape[0] == n_chains

    # this is for silver six only
    #x_init = init[:, :12]
    #u_init = init[:, 12]
    #isomer_init = init[:, 13]

    x_init = init[:, :dim]
    u_init = init[:, dim]
    isomer_init = init[:, dim+1]

    beta = 1 / (kB * T)

    if "dft" in energy_type:
        import gpaw.mpi as mpi
        from flonacomldft.dft_calculator import DFTCalculator
        from flonacomldft.internal_coordinates import Coordinates_mapping
        
        coord_maps = Coordinates_mapping(etype=energy_type)
        calculator = DFTCalculator()

        mpi.world.barrier()

        if dft_folder_name is not None:
            calculator.initialize_calculator(dft_folder_name)
        else:
            calculator.initialize_calculator()
    
        use_dft = True

        xs_dft = []
        us_dft = []
        isomers_dft = []
        inds_dft = []

    else:
        use_dft = False

    #TODO: add emt calculator in the energy computation
    if "emt" in energy_type:
        from flonacomldft.dft_calculator import EMTCalculator
        from flonacomldft.internal_coordinates import Coordinates_mapping

        coord_maps = Coordinates_mapping(etype=energy_type)
        calculator = EMTCalculator()

    #print("Use DFT: ", use_dft)


    if "mlp" in energy_type:
        if(mlp_models is None):
            raise RuntimeError("No MLP model to calculate energy")
        if mixture:
            model_mlp_is0, model_mlp_is1 = mlp_models
        else:
            if(isomer_init.sum()==0):
                model_mlp_is0 = mlp_models
            else:
                model_mlp_is1 = mlp_models

    if return_ratio:
        ratios = []

    if return_proposals:
        xs_props = [init[:, :dim]]

    if mixture:
        weights = []

    xs = []
    us = []
    accs = []
    nlls = []
    isomers = []

    if with_tqdm:
        pbar = tqdm.tqdm(range(n_steps))
    else:
        pbar = range(n_steps)

    timestep_mcmc = []
    
    for dt in pbar:

        if mixture:
            x_new, isomer_new = model.sample(n_chains, return_mus=True)
        else:
            x_new = model.sample(n_chains)
            isomer_new = isomer_init

        x_new = x_new.clone().detach().float()
        isomer_new = isomer_new.clone().detach().float()

        if return_proposals:
            xs_props.append(x_new)

        nll_x = model.nll(x_new)
        nll_x_init = model.nll(x_init)            
       
        if "mlp" in energy_type:
            u_new = torch.zeros((n_chains, 1)).squeeze()

            #print(u_new, isomer_new.bool())

            if mixture:
                #print(u_new[~(isomer_new.bool())], x_new[~(isomer_new.bool())], model_mlp_is0(x_new[~(isomer_new.bool())]))
                u_new[~(isomer_new.bool())] = model_mlp_is0(x_new[~(isomer_new.bool())])
                u_new[isomer_new.bool()] = model_mlp_is1(x_new[isomer_new.bool()])
            else:
                if(isomer_new.sum()==0):
                    u_new = model_mlp_is0[0](x_new)
                else:
                    u_new = model_mlp_is1[0](x_new)

            u_new = u_new.squeeze().float()

        if "dft" in energy_type:
            ind_not_computed = torch.zeros(n_chains) # keeps indices where DFT fails
            ind_dft = torch.zeros(n_chains) # boolean table of where DFT is used
            if energy_type == "dft":
                ind_dft = torch.ones(n_chains)
                u_new = torch.zeros((n_chains))
            else:
                n_dft = int(u_new.shape[0] * frac_dft)
                u_sort, ind_u_sort = u_new.sort()
                for idx in ind_u_sort[:n_dft]:
                    ind_dft[idx] = 1
            #print("DFT idx: ", ind_dft)
            
            # TODO: type of isomer_dft
            for i,flag_dft in enumerate(ind_dft):
                if flag_dft:
                    #try: # Avoid try if possible
                            
                    molecule, logdetjac = coord_maps.build_molecule_from_real_centered(x_new[i].reshape(1, -1), isomer=int(isomer_new[i].item()))
                    u_ = calculator.calculate_potential_energy(
                        molecule, 
                        filename='ag6_{:d}_{:d}_{:d}.out'.format(id_run, dt, i)
                                            )
                    u_new[i] = coord_maps.compute_energy_in_new_frame(u_, logdetjac*(-1))
                    # u_new[i] = torch.tensor(-6.8+torch.rand(1)*0.5)

                    xs_dft.append(x_new[i])
                    us_dft.append(u_new[i])
                    isomers_dft.append(isomer_new[i])

                    #except:
                    #    u_new[i] = 0.
                    #    ind_not_computed[i] = 1

        if "emt" in energy_type:

            u_new = torch.zeros(n_chains)
            for i in range(n_chains):
                molecule, logdetjac = coord_maps.build_molecule_from_real_centered(x_new[i].reshape(1, -1), isomer=int(isomer_new[i].item()))
                u_ = calculator.calculate_potential_energy(molecule)

                u_new[i] = coord_maps.compute_energy_in_new_frame(u_, logdetjac*(-1))

            
        #print(u_new, u_init, nll_x, nll_x_init, isomer_new, isomer_init)

        if use_dft:
            rank = mpi.world.rank
            #print('save time: ', rank)
            if rank == 0:
                timestep_mcmc.append(time.time())
            #print('save time 0 : ', rank)
            mpi.world.barrier()
        else:
            timestep_mcmc.append(time.time())

        ratio = -beta * u_new + nll_x
        ratio += beta * u_init - nll_x_init
        ratio = torch.exp(ratio)

        #print("Ratio: ", ratio)

        if return_ratio:
            ratios.append(torch.min(ratio.clone(), torch.ones_like(ratio)))

        u = torch.rand_like(ratio)

        acc = u < torch.min(ratio, torch.ones_like(ratio))

        if use_dft:
            if ind_not_computed is not None and ind_not_computed.sum() != 0:
                acc[ind_not_computed.bool()] = False

            ind_dft[~acc] = 0
            inds_dft.append(ind_dft.float().clone())

        x_new[~acc] = x_init[~acc]
        u_new[~acc] = u_init[~acc]

        if mixture:
            isomer_new[~acc] = isomer_init[~acc]
        else:
            isomer_new = isomer_init

        
        xs.append(x_new.float().clone())
        us.append(u_new.float().clone())
        accs.append(acc.float().clone())
        nlls.append(nll_x.float().clone())
        isomers.append(isomer_new.float().clone())
        
        x_init = x_new.clone().detach()
        u_init = u_new.clone().detach()
        isomer_init = isomer_new.clone().detach()

        if with_tqdm:
            pbar.set_description(f'acc: {acc.float().mean():.2f}')

        #TODO: Add parameter to save
        #if dt % scheduler == 0:
        print("step: {:d} \t acc: {:0.2f}".format(dt, acc.float().mean()))

        if mixture and update_weigth and dt % scheduler == 0 and dt != 0:
        # if mixture and update_weigth and dt >= scheduler:

            print("dt, isomers shape: ", dt, torch.stack(isomers).shape)
            print("populations window", torch.stack(isomers)[-scheduler:].shape, (~torch.stack(isomers).bool()).float()[-scheduler:].detach().mean(), (torch.stack(isomers).bool()).float()[-scheduler:].detach().mean())
            
            weigths_current_populations = torch.tensor([(~torch.stack(isomers).bool())[-scheduler:].float().mean(), 
                                    (torch.stack(isomers).bool()).float()[-scheduler:].detach().mean()]).float().detach()
            
            print("current weights: ", model.weights)
            print("current population weights: ", weigths_current_populations)

            new_weights = alpha*model.weights.clone() + (1-alpha)*weigths_current_populations.clone()
            
            print("new weights: ", new_weights)

            print(model.weights, new_weights < 0.75, torch.all(new_weights  < 0.75))

            if torch.all(new_weights  < 0.75):

                model.weights = new_weights.clone()
                #model.weights = alpha*model.weights.clone() + (1-alpha)*weigths_current_populations.clone()
                print("updated weights: ", model.weights)

            weights.append(model.weights.clone().float().detach())

    if use_dft:

        timestep_mcmc_min = np.array([0])
        
        ranks = np.arange(mpi.world.size)
        comm = mpi.world.new_communicator(ranks)
        rank = mpi.world.rank
        mpi.world.barrier()

        if rank==0:
            timestep_mcmc_min = np.min(timestep_mcmc).astype(int)
            timestep_mcmc = [t-timestep_mcmc_min for t in timestep_mcmc]
            timestep_mcmc_min = timestep_mcmc_min.reshape(1,)
            timestep_mcmc = np.array(timestep_mcmc)
            
        else:
            timestep_mcmc = np.array([0.]*n_steps)

        comm.broadcast(timestep_mcmc_min, 0)
        comm.broadcast(timestep_mcmc, 0)


        mpi.world.barrier()

        timestep_mcmc_min = timestep_mcmc_min[0]
        timestep_mcmc = timestep_mcmc.tolist()
        timestep_mcmc = [t+timestep_mcmc_min for t in timestep_mcmc]
    
        mpi.world.barrier()

    #print(timestep_mcmc, rank)
    #print(torch.stack(accs), rank)


    to_return = {
        "xs": torch.stack(xs),
        "us": torch.stack(us),
        "accs": torch.stack(accs),
        "isomers": torch.stack(isomers),
        "time_mcmc": timestep_mcmc,
    }

    if return_ratio:
        to_return["ratios"] = torch.stack(ratios)

    if return_proposals:
        to_return["xs_props"] = torch.stack(xs_props)

    if use_dft:
        to_return["inds_dft"] = torch.stack(inds_dft)
        to_return["xs_dft"] = torch.stack(xs_dft)
        to_return["us_dft"] = torch.tensor(us_dft).float().detach()
        to_return['isomers_dft'] = torch.stack(isomers_dft)

    if mixture:
        to_return["weights"] = torch.stack(weights)

    return to_return
