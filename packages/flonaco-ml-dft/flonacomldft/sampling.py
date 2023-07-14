"""
Script with all sampling methods. 

"""

import numpy as np
import torch
import tqdm

from ase.parallel import parprint as print
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
    scheduler = 1,
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
        from flonacomldft.dft_calculator import DFTCalculator
        from flonacomldft.internal_coordinates import Coordinates_mapping
        
        coord_maps = Coordinates_mapping()
        calculator = DFTCalculator()

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

        coord_maps = Coordinates_mapping()
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

    xs = []
    us = []
    accs = []
    nlls = []
    isomers = []

    if with_tqdm:
        pbar = tqdm.tqdm(range(n_steps))
    else:
        pbar = range(n_steps)
    
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
                    try: # Avoid try if possible
                        #TODO: FIX SHAPES OF XS, ZMAT

                        #       zmat, logdetjac = coord_maps.get_internal_from_real_centered(x_new[i].reshape(1, -1), isomer=isomer_new[i].item())
                        #       xyz, logdetjac = coord_maps.get_cartesian_from_internal(zmat[0], logdetjac)
                        #       molecule = coord_maps._build_molecule_from_xyz(xyz)

                        #TODO: BUILD molecule from internal, return logdet

                        ### TODO: LINES THAT ACTUALLY WORK!!!!!!!!!!!!!    
                        molecule, logdetjac = coord_maps.build_molecule_from_real_centered(x_new[i].reshape(1, -1), isomer=int(isomer_new[i].item()))
                        u_ = calculator.calculate_potential_energy(
                            molecule, 
                            filename='ag6_{:d}_{:d}_{:d}.out'.format(id_run, dt, i)
                                            )
                        #print(u_)
                        u_new[i] = coord_maps.compute_energy_in_new_frame(u_, logdetjac*(-1))
                        #print(u_new[i])
                        #u_new[i] = torch.tensor(-6.8+torch.rand(1)*0.5)



                        xs_dft.append(x_new[i])
                        us_dft.append(u_new[i])
                        isomers_dft.append(isomer_new[i])

                    except:
                        u_new[i] = 0.
                        ind_not_computed[i] = 1

        if "emt" in energy_type:

            u_new = torch.zeros(n_chains)
            for i in range(n_chains):
                molecule, logdetjac = coord_maps.build_molecule_from_real_centered(x_new[i].reshape(1, -1), isomer=int(isomer_new[i].item()))
                u_ = calculator.calculate_potential_energy(molecule)

                u_new[i] = coord_maps.compute_energy_in_new_frame(u_, logdetjac*(-1))

            
        #print(u_new, u_init, nll_x, nll_x_init, isomer_new, isomer_init)


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
        if dt % scheduler == 0:
            print("step: {:d} \t acc: {:0.2f}".format(dt, acc.float().mean()))

    to_return = {
        "xs": torch.stack(xs),
        "us": torch.stack(us),
        "accs": torch.stack(accs),
        "isomers": torch.stack(isomers),
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

    return to_return
