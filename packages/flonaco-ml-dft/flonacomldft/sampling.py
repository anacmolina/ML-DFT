"""
Script with all sampling methods. 

"""

# from flonacomldft.utils.data_utils import get_path, load_from_pickle
# from flonacomldft.models import Uncentered_MLP
import numpy as np
import torch
import tqdm

from ase.parallel import parprint as print

# from datetime import datetime
# from flonacomldft.dft_utils import Structure
from flonacomldft.internal_coordinates import Angles_mapping

kb = 8.617333262e-5


def run_metropolis(
    model,
    x_init,
    u_init,
    isomer_init,
    #u_init,
    #x_init,
    #count_init,
    n_chains,
    n_steps,
    n_run="",
    energy_type=None,
    mlps=None,
    mixture=False,
    T=300,
    with_tqdm=False,
):

    assert u_init.shape[0] == n_chains
    assert x_init.shape[0] == n_chains
    assert isomer_init.shape[0] == n_chains

    # print(u_init.shape, x_init.shape, count_init.shape)
    # print('assert pass')

    beta = 1 / (kb * T)

    if ("dft" in energy_type):
        from flonacomldft.dft_utils import DFTCalculator
        from flonacomldft.internal_coordinates import Structure
        
        ag6 = Structure()
        calculator = DFTCalculator()
        calculator.initialize_calculator()
        dft = True

        inds_dft = []
        xs_dft = []
        us_dft = []
        isomers_dft = []
        if (energy_type == "mlp-dft"):        
            energy_type = "mlp"
    else:
        dft = False


    if energy_type == "mlp":
        if(mlps is None):
            raise RuntimeError("No MLP model to calculate energy")
        elif(len(mlps)>1):
            mlp_is1, mlp_is2 = mlps
        elif(len(mlps)==1):
            if(isomer_init.sum()==0):
                mlp_is1 = mlps
            else:
                mlp_is2 = mlps

    #print('init\n',x_init)

    xs = []
    us = []
    accs = []
    nlls = []
    isomers = []
    #xs_prop = []
    #us_prop = []
    
    if with_tqdm:
        pbar = tqdm.tqdm(range(n_steps))
    else:
        pbar = range(n_steps)
    
    for dt in pbar:

        Angles_mapping().inv_mapping(x_init)

        if mixture:
            x, isomer = model.sample(n_chains, return_mus=True)
        else:
            x = model.sample(n_chains)
            isomer = isomer_init

        x = x.clone().detach().float()
        isomer = isomer.clone().detach().float()

        nll_x = model.nll(x)
        nll_x_init = model.nll(x_init)

        Angles_mapping().mapping(x)
        Angles_mapping().mapping(x_init)

        indexes_nc = None
        ind_dft = torch.zeros(x.shape[0])

        if energy_type == "dft":
            U_ = []
            indexes_nc = []

            for i in range(n_chains):
                try:
                    #TODO: Activate this later
                    #u_ = calculator.calculate_potential_energy(ag6.build_molecule(x[i]), file_name='ag6_'+str(n_run)+'_'+str(dt)+'_'+str(i)+'.out')
                    u_ = -6.3*(1+np.random.rand()*0.1)
                    U_.append(u_)

                    xs_dft.append(x[i])
                    us_dft.append(u_)
                    isomers_dft.append(isomer[i])
                except:
                    U_.append(0)
                    indexes_nc.append(i)

            indexes_nc = torch.tensor(indexes_nc)
            ind_dft = torch.ones(x.shape[0])
            U = torch.tensor(U_).float().detach()

            # U = U_.clone().detach()

            #x_prop = x.clone().detach() 
            #u_prop = U.clone().detach()

        elif energy_type == "mlp":

            # print('mlps')

            U_ = torch.zeros((x.shape[0], 1))

            if isomer.sum().int() == isomer.shape[0]:

                #print('mlp_is2')
                model_mlp_is2 = mlp_is2

                U_[isomer.bool()] = model_mlp_is2.predict(x[isomer.bool()])

            if isomer.sum().int() == 0:

                #print('mlp_is1')
                model_mlp_is1 = mlp_is1

                U_[~(isomer.bool())] = model_mlp_is1.predict(x[~(isomer.bool())])

            else:

                mlp_is1, mlp_is2 = mlps[0], mlps[1]

                model_mlp_is1 = mlp_is1
                model_mlp_is2 = mlp_is2

                U_[~(isomer.bool())] = model_mlp_is1.predict(x[~(isomer.bool())])
                U_[isomer.bool()] = model_mlp_is2.predict(x[isomer.bool()])

            U_ = U_.reshape(U_.shape[0]).float()

            if dft:

                # print('mlp-dft')
                n_dft = int(U_.shape[0] * 0.2)

                if n_dft > 0:

                    U_sort, ind_U_sort = U_.sort()

                    U_dft = []
                    indexes_nc = []

                    for i, x_ in enumerate(x[ind_U_sort[:n_dft]]):
                        try:
                            #TODO: Activate this later
                            #u_ = calculator.calculate_potential_energy(ag6.build_molecule(x_), file_name='ag6_'+str(n_run)+'_'+str(dt)+'_'+str(ind_U_sort[:n_dft][i])+'.out')
                            u_ = -6.3*(1+np.random.rand()*0.1)
                            U_dft.append(u_)

                            xs_dft.append(x_)
                            us_dft.append(u_)
                            isomers_dft.append(isomer[ind_U_sort[:n_dft][i]])
                            ind_dft[ind_U_sort[:n_dft][i]] = 1    
                        except:
                            U_dft.append(0)
                            indexes_nc.append(ind_U_sort[:n_dft][i])

                    indexes_nc = torch.tensor(indexes_nc)
                    U_dft = torch.tensor(U_dft).float()

                    U_[ind_U_sort[:n_dft]] = U_dft

            U = U_.clone().float()

            #x_prop = x.clone().detach() 
            #u_prop = U.clone().detach()

        else:
            raise RuntimeError("Unknown method for the energy")

        ratio = -beta * (U) + nll_x
        ratio += beta * u_init - nll_x_init
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc = u < torch.min(ratio, torch.ones_like(ratio))

        if indexes_nc is not None and indexes_nc.shape[0] != 0:
            acc[indexes_nc] = torch.full((1, len(indexes_nc)), False)

        x[~acc] = x_init[~acc]
        U[~acc] = u_init[~acc]
        ind_dft[~acc] = 0

        if mixture:
            isomer[~acc] = isomer_init[~acc]
        else:
            isomer = isomer_init
        
        xs.append(x.float().clone())
        us.append(U.float().clone())
        accs.append(acc.float().clone())
        nlls.append(nll_x.float().clone())
        isomers.append(isomer.float().clone())
        #xs_prop.append(x_prop.float().clone())
        #us_prop.append(u_prop.float().clone())
        if dft:
            inds_dft.append(ind_dft.float().clone())

        x_init = x.clone().detach()
        u_init = U.clone().detach()
        isomer_init = isomer.clone().detach()

        #print("acc: {:0.2f}".format(acc.float().mean()))

    to_return = {
        "xs": torch.stack(xs),
        "us": torch.stack(us),
        "accs": torch.stack(accs),
        "isomers": torch.stack(isomers),
        #"xs_prop": torch.stack(xs_prop),
        #"us_prop": torch.stack(us_prop),
    }
    if dft:
        to_return["inds_dft"] = torch.stack(inds_dft)
        to_return["xs_dft"] = torch.stack(xs_dft)
        to_return["us_dft"] = torch.tensor(us_dft).float().detach()
        to_return['isomers_dft'] = torch.stack(isomers_dft)

    return to_return
