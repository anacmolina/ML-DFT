from ase.parallel import parprint as print

import torch

from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.sampling import run_metropolis
from flonacomldft.models.mixture import Mixture, get_models

from flonacomldft.internal_coordinates import join_data

def Transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def adaptative_sampling(
    flow_init_train,
    flow_init_test,
    n_runs,
    n_chains,
    n_steps,
    energy_type,
    dict_flows_init,
    flow_hyperparams, # dict
    retraining_mlp=False,
    dict_mlps_init=None,
    mlp_hyperparams=None, # dict
    mlp_init_train=None,
    mlp_init_test=None,
):

    """
    function for sampling
    """

    # setting up databases for flows and mlps

    xs_for_flows_train_is0 = flow_init_train.clone()[:,:-1][~flow_init_train[:, 13].bool()]
    xs_for_flows_train_is1 = flow_init_train.clone()[:,:-1][flow_init_train[:, 13].bool()]
 
    xs_for_flows_test_is0 = flow_init_test.clone()[:,:-1][~flow_init_test[:, 13].bool()]
    xs_for_flows_test_is1 = flow_init_test.clone()[:,:-1][flow_init_test[:, 13].bool()]
     
    if retraining_mlp:

        xs_for_mlps_train_is0 = mlp_init_train.clone()[:,:-1][~mlp_init_train[:, 13].bool()]
        xs_for_mlps_train_is1 = mlp_init_train.clone()[:,:-1][mlp_init_train[:, 13].bool()]
 
        xs_for_mlps_test_is0 = mlp_init_test.clone()[:,:-1][~mlp_init_test[:, 13].bool()]
        xs_for_mlps_test_is1 = mlp_init_test.clone()[:,:-1][mlp_init_test[:, 13].bool()]    
    
    dict_flows_training = [dict_flows_init, ]
    dict_mlps_training = [dict_mlps_init, ] #TODO: Not to use if MLPs models None

    mcmc_runs = []

    xs = []
    us = []
    accs = []
    isomers = []
    
    init = flow_init_test[:n_chains]

    for i in range(n_runs):
        
        weights = torch.tensor([0.5, 0.5]).detach()
        mixture = Mixture(get_models(dict_flows_training[i]), weights)

        mcmc_run = run_metropolis(
            model=mixture,
            init=init,
            n_chains=n_chains,
            n_steps=n_steps,
            name_run=i,
            energy_type=energy_type,
            mlp_models=get_models(dict_mlps_training[-1]),
            mixture=True,
            )
        
        mcmc_runs.append(mcmc_run)

        xs.append(mcmc_run["xs"])
        us.append(mcmc_run["us"])
        accs.append(mcmc_run["accs"])
        isomers.append(mcmc_run["isomers"])

        init = join_data(xs[i][-1], us[i][-1], isomers[i][-1])
        
        #"""            
        #### RETRAINING OF THE FLOWS
        chains_flatten = Transpose(
            torch.cat(
                (
                    Transpose(xs[i].clone()),
                    Transpose(us[i].clone().reshape(n_steps, n_chains, 1)),
                    Transpose(isomers[i].clone().reshape(n_steps, n_chains, 1)),
                ),
                dim=0,
            )
        )

        chains_flatten = chains_flatten.reshape(n_steps * n_chains, chains_flatten.shape[-1])

        mask_flow = chains_flatten[:, -1].bool()
        is0_from_chains = chains_flatten[~mask_flow]

        if is0_from_chains.nelement() != 0:

            xs_for_flows_train_is0 = torch.cat(
                (xs_for_flows_train_is0, is0_from_chains)
            )

            dict_new_flow_is0 = train_flow(
                dict_flows_training[i][0]['model'],
                xs_for_flows_train_is0,
                xs_for_flows_test_is0,
                **flow_hyperparams[0],
                )
            
        else:
            dict_new_flow_is0 = dict_flows_training[i][0]

        is1_from_chains = chains_flatten[mask_flow]

        if is1_from_chains.nelement() != 0:

            xs_for_flows_train_is1 = torch.cat(
                (xs_for_flows_train_is1, is1_from_chains)
            )  
            
            dict_new_flow_is1 = train_flow(
                dict_flows_training[i][1]['model'],
                xs_for_flows_train_is1,
                xs_for_flows_test_is1,
                **flow_hyperparams[1],)

        else:
            dict_new_flow_is1 = dict_flows_training[i][1]
        #"""

        #### RETRAINING OF THE MLPS
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

            print(xs_for_mlps_train_is0, configs_dft_flatten[~mask_mlp])

            xs_for_mlps_train_is0 = torch.cat((xs_for_mlps_train_is0, configs_dft_flatten[~mask_mlp]))
            xs_for_mlps_train_is1 = torch.cat((xs_for_mlps_train_is1, configs_dft_flatten[mask_mlp]))

            from flonacomldft.train_mlp_from_data import train_mlp

            mlp_is0 = train_mlp(dict_mlps_training[-1][0]['model'], 
                                xs_for_mlps_train_is0,  
                                xs_for_mlps_test_is0, 
                                **mlp_hyperparams[0],
                                with_tqdm=True)

            mlp_is1 = train_mlp(dict_mlps_training[-1][1]['model'], 
                                xs_for_mlps_train_is1,  
                                xs_for_mlps_test_is1, 
                                **mlp_hyperparams[1],
                                with_tqdm=True)

            dict_mlps_training.append([mlp_is0, mlp_is1])
        

        dict_flows_training.append([dict_new_flow_is0, dict_new_flow_is1])

    results = {
        'mcmc': mcmc_runs,
        'xs': xs,
        'us': us,
        'accs': accs,
        'isomers': isomers,
        'flows_training': dict_flows_training,
    }

    if retraining_mlp:
        results['mlps_training'] = dict_mlps_training,
    
    
    return results