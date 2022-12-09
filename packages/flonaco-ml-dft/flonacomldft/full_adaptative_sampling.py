from ase.parallel import parprint as print

import torch
import numpy as np
import matplotlib.pyplot as plt

from flonacomldft.internal_coordinates import Angles_mapping
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.sampling import run_metropolis
from flonacomldft.mixture import Mixture, get_models, get_models_mlp

# TODO: add the losses for testint the models

def Transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

def adaptative_sampling(
    xs_md_init,
    us_md_init,
    isomers_md_init,
    xs_dft_init,
    us_dft_init,
    isomers_dft_init,
    n_runs,
    n_chains,
    n_steps,
    energy_type,
    dict_flows_init,
    flow_hyperparams, # dict
    dict_mlps_init,
    mlp_hyperparams, # dict
    retraining_mlp=False
):

    """
    function for sampling
    """

    # setting up databases for flows and mlps

    xs_for_flows_is1 = xs_md_init[~isomers_md_init.bool()]
    xs_for_flows_is2 = xs_md_init[isomers_md_init.bool()]

    us_for_flows_is1 = us_md_init[~isomers_md_init.bool()]
    us_for_flows_is2 = us_md_init[isomers_md_init.bool()]

    if retraining_mlp:

        xs_for_mlps_is1 = xs_dft_init[~isomers_dft_init.bool()]
        xs_for_mlps_is2 = xs_dft_init[isomers_dft_init.bool()]

        us_for_mlps_is1 = us_dft_init[~isomers_dft_init.bool()]
        us_for_mlps_is2 = us_dft_init[isomers_dft_init.bool()]

    dic_flows_training = [dict_flows_init, ]
    dic_mlps_training = [dict_mlps_init, ]

    mcmc = []

    xs = []
    us = []
    accs = []
    isomers = []
    
    x_init = xs_md_init[:n_chains]
    u_init = us_md_init[:n_chains]
    isomer_init = isomers_md_init[:n_chains]

    for i in range(n_runs):
        weights = torch.tensor([0.5, 0.5]).detach()
        mixture = Mixture(get_models(dic_flows_training[i]), weights)

        mcmc_run = run_metropolis(
        model=mixture,
        x_init=x_init,
        u_init=u_init,
        isomer_init=isomer_init,
        n_chains=n_chains,
        n_steps=n_steps,
        n_run=i,
        energy_type=energy_type,
        #mlps=get_models(dic_mlps_training[-1]),
        mlps=get_models_mlp(dic_mlps_training[-1]),
        mixture=True,
        )
        
        mcmc.append(mcmc_run)

        xs.append(mcmc_run["xs"])
        us.append(mcmc_run["us"])
        accs.append(mcmc_run["accs"])
        isomers.append(mcmc_run["isomers"])

        u_init = us[i][-1]
        x_init = xs[i][-1]
        isomer_init = isomers[i][-1]

        chains_flatten = Transpose(
            torch.cat(
                (
                    Transpose(xs[i].clone()),
                    Transpose(isomers[i].clone().reshape(n_steps, n_chains, 1)),
                ),
                dim=0,
            )
        )

        chains_flatten = chains_flatten.reshape(n_steps * n_chains, chains_flatten.shape[-1])

        mask_flow = chains_flatten[:, -1] == 1
        is1_from_chains = chains_flatten[~mask_flow][:, :-1]

        if is1_from_chains.nelement() != 0:
            xs_for_flows_is1 = torch.cat(
                (xs_for_flows_is1, is1_from_chains)
            )

            Angles_mapping().inv_mapping(xs_for_flows_is1)

            dic_new_flow_is1 = train_flow(
                dic_flows_training[i][0]['model'],
                xs_for_flows_is1,
                **flow_hyperparams,)

            Angles_mapping().mapping(xs_for_flows_is1)
        else:
            dic_new_flow_is1 = dic_flows_training[i][0]

        is2_from_chains = chains_flatten[mask_flow][:, :-1]

        if is2_from_chains.nelement() != 0:
            xs_for_flows_is2 = torch.cat(
                (xs_for_flows_is2, is2_from_chains)
            )

            Angles_mapping().inv_mapping(xs_for_flows_is2)    

            dic_new_flow_is2 = train_flow(
                dic_flows_training[i][1]['model'],
                xs_for_flows_is2,
                **flow_hyperparams,)
            Angles_mapping().mapping(xs_for_flows_is2)
        else:
            dic_new_flow_is2 = dic_flows_training[i][1]
        
        # retrain MLPs
        if retraining_mlp and ('dft' in energy_type):

            xs_dft = mcmc_run['xs_dft']
            us_dft = mcmc_run['us_dft']
            isomers_dft = mcmc_run['isomers_dft']

            configs_dft_flatten = Transpose(
                torch.cat(
                    (
                        Transpose(xs_dft),
                        us_dft.reshape(1, -1),
                        isomers_dft.reshape(1, -1),
                    ),
                    dim=0,
                )
            )

            mask_mlp = configs_dft_flatten[:, -1].bool()

            xs_for_mlps_is1 = torch.cat((xs_for_mlps_is1, configs_dft_flatten[~mask_mlp][:, :-2]))
            us_for_mlps_is1 = torch.cat((us_for_mlps_is1, configs_dft_flatten[~mask_mlp][:, -2]))

            xs_for_mlps_is2 = torch.cat((xs_for_mlps_is2, configs_dft_flatten[mask_mlp][:, :-2]))
            us_for_mlps_is2 = torch.cat((us_for_mlps_is2, configs_dft_flatten[mask_mlp][:, -2]))

            from flonacomldft.train_mlp_from_data import train_mlp

            #mlp_is1 = train_mlp(dic_mlps_training[-1][0]['model'], xs_for_mlps_is1, us_for_mlps_is1, xs_for_mlps_is1, us_for_mlps_is1, **mlp_hyperparams)
            #mlp_is2 = train_mlp(dic_mlps_training[-1][1]['model'], xs_for_mlps_is2, us_for_mlps_is2, xs_for_mlps_is2, us_for_mlps_is2, **mlp_hyperparams)

            mlp_is1 = train_mlp(dic_mlps_training[-1][0], xs_for_mlps_is1, us_for_mlps_is1, xs_for_mlps_is1, us_for_mlps_is1, **mlp_hyperparams)
            mlp_is2 = train_mlp(dic_mlps_training[-1][1], xs_for_mlps_is2, us_for_mlps_is2, xs_for_mlps_is2, us_for_mlps_is2, **mlp_hyperparams)

            dic_flows_training.append([mlp_is1, mlp_is2])

        dic_flows_training.append([dic_new_flow_is1, dic_new_flow_is2])

    results = {
        'mcmc': mcmc,
        'xs': xs,
        'us': us,
        'accs': accs,
        'isomers': isomers,
        'flows_training': dic_flows_training,
    }

    if retraining_mlp:
        results['mlps_training'] = dic_mlps_training,
    
    
    return results