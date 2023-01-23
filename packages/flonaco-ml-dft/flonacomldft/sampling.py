"""
Script with all sampling methods. 

"""

import numpy as np
import torch
import tqdm

from ase.parallel import parprint as print


kb = 8.617333262e-5

def run_metropolis(
    model,
    x_init,
    u_init,
    isomer_init,
    n_chains,
    n_steps,
    n_run="",
    energy_type=None,
    frac_dft=0.2,
    mlp_models=None,
    mixture=False,
    T=300,
    with_tqdm=False,
):

    assert u_init.shape[0] == n_chains
    assert x_init.shape[0] == n_chains
    assert isomer_init.shape[0] == n_chains

    # print(u_init.shape, x_init.shape, isomer_init.shape)
    # print('assert pass')

    beta = 1 / (kb * T)

    if "dft" in energy_type:
        from flonacomldft.dft_calculator import DFTCalculator
        from flonacomldft.internal_coordinates import Coordinates_mapping
        
        coord_maps = Coordinates_mapping()
        calculator = DFTCalculator()
        calculator.initialize_calculator()
        use_dft = True

        xs_dft = []
        us_dft = []
        isomers_dft = []
        inds_dft = []

    else:
        # mlp_dft = False
        use_dft = False

    #print("Use DFT: ", use_dft)


    if "mlp" in energy_type:
        if(mlp_models is None):
            raise RuntimeError("No MLP model to calculate energy")
        if mixture:
            model_mlp_is1, model_mlp_is2 = mlp_models
        else:
            if(isomer_init.sum()==0):
                model_mlp_is1 = mlp_models
            else:
                model_mlp_is2 = mlp_models

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

        nll_x = model.nll(model.rad_center(x_new, isomer_new))
        nll_x_init = model.nll(model.rad_center(x_init, isomer_init))
       
        if "mlp" in energy_type:
            u_new = torch.zeros((n_chains, 1))
            # Marylou: I suspect the line below are unnecessary - commented for now
            # if count.sum().int() == count.shape[0]:
            #     u_new[count.bool()] = model_mlp_is2.predict(x[count.bool()])
            # elif count.sum().int() == 0:
            #     u_new[~(count.bool())] = model_mlp_is1.predict(x[~(count.bool())])
            # else:

            if mixture:
                u_new[~(isomer_new.bool())] = model_mlp_is1.predict(x_new[~(isomer_new.bool())])
                u_new[isomer_new.bool()] = model_mlp_is2.predict(x_new[isomer_new.bool()])
            else:
                if(isomer_new.sum()==0):
                    u_new = model_mlp_is1.predict(x_new)
                else:
                    u_new = model_mlp_is2.predict(x_new)

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
                    try:
                        #TODO: CHANGE THIS
                        u_ = calculator.calculate_potential_energy(
                            coord_maps.get_molecule_from_internal(x_new[i]), 
                            filename='ag6_'+str(n_run)+'_'+str(dt)+'_'+str(i)+'.out'
                                            )
                        u_new[i] = u_ - logdetjac_to_xyz(x_new[i]) / beta
                        #u_new[i] = torch.tensor(-6.3*(1+np.random.rand()*0.1))

                        xs_dft.append(x_new[i])
                        #us_dft.append(u_) #TODO: ALERT! Possible BUG, ALERT
                        us_dft.append(u_new[i])
                        isomers_dft.append(isomer_new[i])
                    except:
                        u_new[i] = 0.
                        ind_not_computed[i] = 1

        ratio = -beta * u_new + nll_x
        ratio += beta * u_init - nll_x_init
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)

        acc = u < torch.min(ratio, torch.ones_like(ratio))

        if use_dft:
            if ind_not_computed is not None and ind_not_computed.sum() != 0:
                acc[ind_not_computed.bool()] = torch.full((1, len(ind_not_computed)), False)

            ind_dft[~acc] = 0
            inds_dft.append(ind_dft.float().clone())

        x_new[~acc] = x_init[~acc]
        u_new[~acc] = u_init[~acc]

        if mixture:
            isomer_new[~acc] = isomer_init[~acc]
        else:
            isomer_new = isomer_init

        #print('accs: {:2f}'.format(acc.float().mean()))
        
        xs.append(x_new.float().clone())
        us.append(u_new.float().clone())
        accs.append(acc.float().clone())
        nlls.append(nll_x.float().clone())
        isomers.append(isomer_new.float().clone())
        
        #if use_dft:
        #    inds_dft.append(ind_dft.float().clone())

        x_init = x_new.clone().detach()
        u_init = u_new.clone().detach()
        isomer_init = isomer_new.clone().detach()

        if with_tqdm:
            pbar.set_description(f'acc: {acc.float().mean():.2f}')


        #print("acc: {:0.2f}".format(acc.float().mean()))

    to_return = {
        "xs": torch.stack(xs),
        "us": torch.stack(us),
        "accs": torch.stack(accs),
        "isomers": torch.stack(isomers),

    }
    if use_dft:
        #TODO: empty lists BUG
        to_return["inds_dft"] = torch.stack(inds_dft)
        to_return["xs_dft"] = torch.stack(xs_dft)
        to_return["us_dft"] = torch.tensor(us_dft).float().detach()
        to_return['isomers_dft'] = torch.stack(isomers_dft)

    return to_return
