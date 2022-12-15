from ase.parallel import parprint as print

import torch

from flonacomldft.internal_coordinates import Angles_mapping
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.sampling import run_metropolis
from flonacomldft.mixture import Mixture, get_models

def Transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

def adaptative_sampling(
    x_init,
    u_init,
    isomer_init,
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

    xs_for_flows_train_is1, xs_for_flows_test_is1 = dict_flows_init[0]['dataset']
    xs_for_flows_train_is2, xs_for_flows_test_is2 = dict_flows_init[1]['dataset']

    Angles_mapping().mapping(xs_for_flows_train_is1)
    Angles_mapping().mapping(xs_for_flows_test_is1)
    Angles_mapping().mapping(xs_for_flows_train_is2)
    Angles_mapping().mapping(xs_for_flows_test_is2)
        

    if retraining_mlp:

        xs_for_mlps_train_is1, xs_for_mlps_test_is1, us_for_mlps_train_is1, us_for_mlps_test_is1 = dict_mlps_init[0]['dataset']
        xs_for_mlps_train_is2, xs_for_mlps_test_is2, us_for_mlps_train_is2, us_for_mlps_test_is2 = dict_mlps_init[1]['dataset']

        
    dict_flows_training = [dict_flows_init, ]
    dict_mlps_training = [dict_mlps_init, ]

    mcmc = []

    xs = []
    us = []
    accs = []
    isomers = []
    
    for i in range(n_runs):

        
        weights = torch.tensor([0.5, 0.5]).detach()
        mixture = Mixture(get_models(dict_flows_training[i]), weights)

        mcmc_run = run_metropolis(
        model=mixture,
        x_init=x_init,
        u_init=u_init,
        isomer_init=isomer_init,
        n_chains=n_chains,
        n_steps=n_steps,
        n_run=i,
        energy_type=energy_type,
        mlps=get_models(dict_mlps_training[-1]),
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
            xs_for_flows_train_is1 = torch.cat(
                (xs_for_flows_train_is1, is1_from_chains)
            )

            Angles_mapping().inv_mapping(xs_for_flows_train_is1)
            Angles_mapping().inv_mapping(xs_for_flows_test_is1)

            dict_new_flow_is1 = train_flow(
                dict_flows_training[i][0]['model'],
                xs_for_flows_train_is1,
                xs_for_flows_test_is1,
                **flow_hyperparams[0],)

            Angles_mapping().mapping(xs_for_flows_train_is1)
            Angles_mapping().mapping(xs_for_flows_test_is1)
        else:
            dict_new_flow_is1 = dict_flows_training[i][0]

        is2_from_chains = chains_flatten[mask_flow][:, :-1]

        if is2_from_chains.nelement() != 0:
            xs_for_flows_train_is2 = torch.cat(
                (xs_for_flows_train_is2, is2_from_chains)
            )

            Angles_mapping().inv_mapping(xs_for_flows_train_is2)    
            Angles_mapping().inv_mapping(xs_for_flows_test_is2)    

            dict_new_flow_is2 = train_flow(
                dict_flows_training[i][1]['model'],
                xs_for_flows_train_is2,
                xs_for_mlps_test_is2,
                **flow_hyperparams[1],)
            Angles_mapping().mapping(xs_for_flows_train_is2)
            Angles_mapping().mapping(xs_for_flows_test_is2)
        else:
            dict_new_flow_is2 = dict_flows_training[i][1]
        
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

            xs_for_mlps_train_is1 = torch.cat((xs_for_mlps_train_is1, configs_dft_flatten[~mask_mlp][:, :-2]))
            us_for_mlps_train_is1 = torch.cat((us_for_mlps_train_is1, configs_dft_flatten[~mask_mlp][:, -2]))

            xs_for_mlps_train_is2 = torch.cat((xs_for_mlps_train_is2, configs_dft_flatten[mask_mlp][:, :-2]))
            us_for_mlps_train_is2 = torch.cat((us_for_mlps_train_is2, configs_dft_flatten[mask_mlp][:, -2]))

            from flonacomldft.train_mlp_from_data import train_mlp

            mlp_is1 = train_mlp(dict_mlps_training[-1][0]['model'], xs_for_mlps_train_is1, xs_for_mlps_test_is1, us_for_mlps_train_is1, us_for_mlps_test_is1, **mlp_hyperparams[0])
            mlp_is2 = train_mlp(dict_mlps_training[-1][1]['model'], xs_for_mlps_train_is2, xs_for_mlps_test_is2, us_for_mlps_train_is2, us_for_mlps_test_is2, **mlp_hyperparams[1])

            dict_mlps_training.append([mlp_is1, mlp_is2])

        dict_flows_training.append([dict_new_flow_is1, dict_new_flow_is2])

    results = {
        'mcmc': mcmc,
        'xs': xs,
        'us': us,
        'accs': accs,
        'isomers': isomers,
        'flows_training': dict_flows_training,
    }

    if retraining_mlp:
        results['mlps_training'] = dict_mlps_training,
    
    
    return results