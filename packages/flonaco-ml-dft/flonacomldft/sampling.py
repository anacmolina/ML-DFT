"""
Script with all sampling methods. 

"""

from typing_extensions import runtime
from flonacomldft.utils.data_utils import get_path, load_from_pickle
#from flonacomldft.models import Uncentered_MLP
import numpy as np
import torch

from ase.parallel import parprint as print

# from datetime import datetime
from flonacomldft.dft_utils import Structure
from flonacomldft.internal_coordinates import Angles_mapping


"""
def run_metropolis(model, target, x_init, n_steps):
    xs = []
    accs = []

    for dt in range(n_steps):
        x = model.sample(x_init.shape[0])
        ratio = - target.beta * target.U(x) + model.nll(x)
        ratio += target.beta * target.U(x_init) - model.nll(x_init)
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc = u < torch.min(ratio, torch.ones_like(ratio))
        x[~acc] = x_init[~acc]
        xs.append(x.clone())
        accs.append(acc)
        x_init = x.clone()

    return torch.stack(xs), torch.stack(accs)
"""

#TODO: Remove this later
#torch.manual_seed(36)

def run_metropolis(
    model,
    u_init,
    x_init,
    count_init,
    n_chains,
    n_steps,
    energy_type=None,
    mlps=None,
    mixture=False,
):

    assert u_init.shape[0] == n_chains
    assert x_init.shape[0] == n_chains
    assert count_init.shape[0] == n_chains

    # print(u_init.shape, x_init.shape, count_init.shape)
    # print('assert pass')

    if energy_type == "mlp-dft":
        mlp_dft = True
        energy_type = "mlp"
    else:
        mlp_dft = False

    if energy_type == "mlp":
        if(mlps is None):
            raise RuntimeError("No MLP model to calculate energy")
        elif(len(mlps)>1):
            mlp_is1, mlp_is2 = mlps
        elif(len(mlps)==1):
            if(count_init.sum()==0):
                mlp_is1 = mlps
            else:
                mlp_is2 = mlps

    #print('init\n',x_init)

    xs = []
    accs = []
    us = []
    us_p = []
    nlls = []
    counts = []
    ind_dft = []

    T = 300
    kb = 8.617333262e-5
    beta = 1 / (kb * T)

    angles_mapping = Angles_mapping()

    for dt in range(n_steps):

        angles_mapping.inv_mapping(x_init)

        if mixture:
            x, count = model.sample(n_chains, return_mus=True)
        else:
            x = model.sample(n_chains)
            count = count_init

        x = x.clone().detach().float()
        count = count.clone().detach().float()

        nll_x = model.nll(x)
        nll_x_init = model.nll(x_init)

        angles_mapping.mapping(x)
        angles_mapping.mapping(x_init)

        indexes_nc = None
        ind_dft_ = torch.zeros(x.shape[0])

        if energy_type == "dft":

            # print('dft')

            ag6 = Structure()

            U_ = []
            indexes_nc = []

            for i in range(n_chains):
                try:
                    #ag6.calculate_potential_energy(
                    #    x[i], txt="ag6_" + str(i) + "_" + str(dt) + ".out"
                    #)
                    #U_.append(ag6.potential_energy)
                    U_.append(-6.3*(1+np.random.rand()*0.1))
                except:
                    U_.append(0)
                    indexes_nc.append(i)

            indexes_nc = torch.tensor(indexes_nc)
            ind_dft_ = torch.ones(x.shape[0])
            U_ = torch.tensor(U_).float()
            U = U_.clone().detach()

        elif energy_type == "mlp":

            # print('mlps')

            U_ = torch.zeros((x.shape[0], 1))

            if count.sum().int() == count.shape[0]:

                #print('mlp_is2')
                model_mlp_is2 = mlp_is2

                U_[count.bool()] = model_mlp_is2.predict(x[count.bool()])

            if count.sum().int() == 0:

                #print('mlp_is1')
                model_mlp_is1 = mlp_is1

                U_[~(count.bool())] = model_mlp_is1.predict(x[~(count.bool())])

            else:

                #print('mlp_is1_is2')

                mlp_is1, mlp_is2 = mlps

                model_mlp_is1 = mlp_is1
                model_mlp_is2 = mlp_is2

                U_[~(count.bool())] = model_mlp_is1.predict(x[~(count.bool())])
                U_[count.bool()] = model_mlp_is2.predict(x[count.bool()])

            U_ = U_.reshape(U_.shape[0]).float()

            if mlp_dft:

                # print('mlp-dft')
                n_dft = int(U_.shape[0] * 0.2)

                if n_dft > 0:

                    U_sort, ind_U_sort = U_.sort()

                    U_dft = []
                    indexes_nc = []

                    ag6 = Structure()

                    for i, x_ in enumerate(x[ind_U_sort[:n_dft]]):
                        try:
                            #ag6.calculate_potential_energy(x_)
                            #U_dft.append(ag6.potential_energy)
                            U_dft.append(-6.3*(1+np.random.rand()*0.1))
                            ind_dft_[ind_U_sort[:n_dft][i]] = 1
                        except:
                            U_dft.append(0)
                            indexes_nc.append(ind_U_sort[:n_dft][i])

                    indexes_nc = torch.tensor(indexes_nc)
                    U_dft = torch.tensor(U_dft).float()

                    U_[ind_U_sort[:n_dft]] = U_dft

            U = U_.clone().float()

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
        ind_dft_[~acc] = 0

        if mixture:
            count[~acc] = count_init[~acc]
        else:
            count = count_init
        
        xs.append(x.float().clone())
        accs.append(acc.float().clone())
        us.append(U.float().clone())
        us_p.append(U_.float().clone())
        nlls.append(nll_x.float().clone())
        counts.append(count.float().clone())
        ind_dft.append(ind_dft_.float().clone())

        x_init = x.clone().detach()
        u_init = U.clone().detach()
        count_init = count.clone().detach()

        #print("acc: {:0.2f}".format(acc.float().mean()))

    to_return = {
        "xs": torch.stack(xs),
        "accs": torch.stack(accs),
        "us": torch.stack(us),
        "us_p": torch.stack(us_p),
        "counts": torch.stack(counts),
        "ind_dft": torch.stack(ind_dft),
    }

    return to_return
