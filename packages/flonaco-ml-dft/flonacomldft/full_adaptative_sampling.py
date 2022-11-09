from ase.parallel import parprint as print

import torch
import numpy as np

from flonacomldft.internal_coordinates import Angles_mapping
from flonacomldft.mixture import Mixture, get_models
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.sampling import run_metropolis

def T(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

def adaptative_sampling(x_init, u_init, count_init, n_runs, n_chains, n_steps, energy_type, flow_models, mlp_models):

    flow = [
        flow_models,
    ]
    
    mlps = [
        mlp_models,
    ]

    flow_train = [
    ]

    xs_acc = []
    us_acc = []
    accs_s = []
    cs_acc = []

    USE_DFT_ENERGIES = False
    if energy_type == "dft" or energy_type == "mlp-dft":
        USE_DFT_ENERGIES = True

    for i in range(n_runs):

        weights = torch.tensor([0.5, 0.5]).detach()
        mixture = Mixture(flow[i], weights)        

        _ = run_metropolis(
        model=mixture,
        #u_init=us_acc[i][-1],
        #x_init=xs_acc[i][-1],
        #count_init=cs_acc[i][-1],
        u_init=u_init,
        x_init=x_init,
        count_init=count_init,
        n_chains=n_chains,
        n_steps=n_steps,
        energy_type=energy_type,
        mlps=mlps[0],
        mixture=True,
        )

        xs_acc.append(_["xs"])
        us_acc.append(_["us"])
        accs_s.append(_["accs"])
        cs_acc.append(_["counts"])

        u_init = us_acc[i][-1]
        x_init = xs_acc[i][-1]
        count_init = cs_acc[i][-1]

        xs = xs_acc[i].clone()
        us = us_acc[i].clone()
        cs = cs_acc[i].clone()

        print(i)
        data_for_flows = T(
            torch.cat(
                (
                    #T(xs_acc[i]),
                    #T(cs_acc[i].reshape(n_sts, n_chains, 1)),
                    T(xs),
                    T(cs.reshape(n_steps, n_chains, 1)),
                ),
                dim=0,
            )
        )

        
        data_for_flows = data_for_flows.reshape(n_steps * n_chains, data_for_flows.shape[-1])
        #data_for_flows = data_for_flows.unique(dim=0)

        mask_flow = data_for_flows[:, -1] == 1
        is1_prop = data_for_flows[~mask_flow][:, :-1]
        is2_prop = data_for_flows[mask_flow][:, :-1]

        # print(data_for_flows)
        #del data_for_flows
        #print(is1_prop)
        #print(is2_prop)

        M = Angles_mapping()
        M.inv_mapping(is1_prop)
        M.inv_mapping(is2_prop)

        if is1_prop.nelement() != 0:
            new_flow_is1 = train_flow(
                flow[i][0],
                is1_prop,
                n_iter=100,
                lr=5e-3,
                bs=100,
                use_scheduler=False,
                step_schedule=100,
                args_loss={"type": "fwd", "samp": "direct"},
                save_splits=10,
                grad_clip=1e4,)
        else:
            new_flow_is1 = flow_train[i][0]

        if is2_prop.nelement() != 0:
            new_flow_is2 = train_flow(
                flow[i][1],
                is2_prop,
                n_iter=100,
                lr=5e-3,
                bs=100,
                use_scheduler=False,
                step_schedule=100,
                args_loss={"type": "fwd", "samp": "direct"},
                save_splits=10,
                grad_clip=1e4,)
        else:
            new_flow_is2 = flow_train[i][1]

        M.mapping(is1_prop)
        M.mapping(is2_prop)

        # retrain MLPs
        if USE_DFT_ENERGIES:

            data_for_mlp = T(
                torch.cat(
                    (
                        T(xs),
                        T(us.reshape(n_steps, n_chains, 1)),
                        T(cs.reshape(n_steps, n_chains, 1)),
                    ),
                    dim=0,
                )
            )

            #print(data_for_mlp, _['ind_dft'])
            data_for_mlp = data_for_mlp[_['ind_dft'].bool()]
            mask_mlp = data_for_mlp[:, -1] == 1

            is1_prop_dft = data_for_mlp[~mask_mlp]
            is2_prop_dft = data_for_mlp[mask_mlp]

            #print(data_for_mlp)
            #print(is1_prop_dft)
            #print(is2_prop_dft)

            if energy_type == "dft":
                # use all # all index must be  ind_dft == 1
                data_ex = data_for_mlp.reshape(n_steps * n_chains, data_for_mlp.shape[-1])

            elif energy_type == "mlp-dft":
                # use only ind_dft
                print("hi :(, T_T")  # _["ind_dft"])

        flow_train.append([new_flow_is1, new_flow_is2])
        flow.append(get_models(flow_train[i]))
        
    #TODO: free space, delete variable

    results = {
        'xs': xs_acc,
        'us': us_acc,
        'acc': accs_s,
        'counts': cs_acc,
    }

    return results