from ase.parallel import parprint as print
from ase.visualize.plot import plot_atoms

import torch
import numpy as np
import matplotlib.pyplot as plt

from flonacomldft.internal_coordinates import Angles_mapping
from flonacomldft.mixture import Mixture, get_models
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.sampling import run_metropolis
from flonacomldft.internal_coordinates import Structure

def Transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

def adaptative_sampling(x, u, count, n_runs, n_chains, n_steps, energy_type,
                       dic_flow_training_init, mlp_models):
    """
    function for sampling
    """

    dic_flow_trainings = [dic_flow_training_init,
    ]

    # keeping track of flows separately is redundant
    flows = [
        get_models(dic_flow_trainings[0]),
    ]
    
    mlps = [
        mlp_models,
    ]

    xs_acc = []
    us_acc = []
    accs_s = []
    cs_acc = []

    USE_DFT_ENERGIES = False
    if "dft" in energy_type:
        USE_DFT_ENERGIES = True

    x_init = x[:n_chains]
    u_init = u[:n_chains]
    count_init = count[:n_chains]

    x_flow_is1 = x[~count.bool()]
    x_flow_is2 = x[count.bool()]

    x_mlp_is1 = x[~count.bool()]
    u_mlp_is1 = u[~count.bool()]
    
    x_mlp_is2 = x[count.bool()]
    u_mlp_is2 = u[count.bool()]

    #ag6 = Structure()
    #plot_atoms(ag6.build_molecule(x_flow_is1[0]))
    #plt.show(block=False)
    #plot_atoms(ag6.build_molecule(x_flow_is2[0]))
    #plt.show(block=False)

    for i in range(n_runs):

        weights = torch.tensor([0.5, 0.5]).detach()
        mixture = Mixture(flows[i], weights)        

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
        n_run=i,
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

        print("shape chains: ", xs.shape)
        data_for_flows = Transpose(
            torch.cat(
                (
                    #T(xs_acc[i]),
                    #T(cs_acc[i].reshape(n_sts, n_chains, 1)),
                    Transpose(xs),
                    Transpose(cs.reshape(n_steps, n_chains, 1)),
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
        print("x_flow shape: ", x_flow_is1.shape)
        print("is1_prop: ", is1_prop.shape)
        x_flow_is1 = torch.cat((x_flow_is1, is1_prop))
        print("x_flow shape after: ", x_flow_is1.shape)
        #print(is1_prop)
        #print(is2_prop)

        angles_mapping = Angles_mapping()
        angles_mapping.inv_mapping(x_flow_is1)
        angles_mapping.inv_mapping(x_flow_is2)

        if is1_prop.nelement() != 0:
            dic_new_flow_is1 = train_flow(
                flows[i][0],
                x_flow_is1,
                n_iter=100,
                lr=5e-3,
                use_scheduler=False,
                step_schedule=100,
                save_splits=10,
                grad_clip=1e4,)
        else:
            dic_new_flow_is1 = dic_flow_trainings[i][0]

        if is2_prop.nelement() != 0:
            dic_new_flow_is2 = train_flow(
                flows[i][1],
                x_flow_is2,
                n_iter=100,
                lr=5e-3,
                use_scheduler=False,
                step_schedule=100,
                save_splits=10,
                grad_clip=1e4,)
        else:
            dic_new_flow_is2 = dic_flow_trainings[i][1]

        angles_mapping.mapping(x_flow_is1)
        angles_mapping.mapping(x_flow_is2)

        # retrain MLPs
        if USE_DFT_ENERGIES:

            data_for_mlp = Transpose(
                torch.cat(
                    (
                        Transpose(xs),
                        Transpose(us.reshape(n_steps, n_chains, 1)),
                        Transpose(cs.reshape(n_steps, n_chains, 1)),
                    ),
                    dim=0,
                )
            )

            #print(data_for_mlp, _['ind_dft'])
            data_for_mlp = data_for_mlp[_['inds_dft'].bool()]
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

        dic_flow_trainings.append([dic_new_flow_is1, dic_new_flow_is2])
        flows.append(get_models(dic_flow_trainings[i]))
        
    #TODO: free space, delete variable

    results = {
        'xs': xs_acc,
        'us': us_acc,
        'acc': accs_s,
        'counts': cs_acc,
    }

    all_models = {'flows': dic_flow_trainings,
                'mlps': mlps
    }

    return results, all_models