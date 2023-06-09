import torch
import copy
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.sampling import run_metropolis
from flonacomldft.models.mixture import Mixture, get_models
from flonacomldft.internal_coordinates import join_data

def get_weights(isomers):

    labels = torch.unique(isomers)

    weights = torch.zeros_like(labels, dtype=torch.float)

    for i in labels.int():
        label_size = torch.ones_like(isomers[isomers == i].flatten()).sum().item()
        weights[i] = label_size/isomers.flatten().shape[0]

    return weights

def Transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

def adaptive_sampling(
        flow_init_train,
        flow_init_test,
        init_mcmc,
        dict_flows_init,
        dict_mlps_init,
        flow_hyperparams,
        n_chains,
        n_steps,
        n_runs,
        dim=12,
        energy_type='mlp',
        mixture=False,
        T=300,
        reweighting=False,
        weights=None,
        dft_folder_name=None
        ):

    modes = torch.unique(flow_init_train[:, dim+1])

    xs_for_flows_train = [ flow_init_train[flow_init_train[:, dim+1] == mode] for mode in modes ]
    xs_for_flows_test = [ flow_init_test[flow_init_test[:, dim+1] == mode] for mode in modes ]

    dict_flows_training = [copy.deepcopy(dict_flows_init), ]

    if 'mlp' in energy_type:
        mlp_models = get_models(dict_mlps_init)
    else:
        mlp_models = [None]

    mcmc_runs = []

    xs = []
    us = []
    accs = []
    isomers = []
    
    init = init_mcmc

    if weights is None:
        weights = torch.tensor([0.5]*len(modes)).detach()
    else:
        weights = torch.tensor(weights).detach()

    for i in range(n_runs):

        if mixture:
            if reweighting and i>0:
                weights = get_weights(isomers[i-1])

            model = Mixture(get_models(dict_flows_training[i]), weights)
        else:
            model = get_models(dict_flows_training[i])[0]

        mcmc_run = run_metropolis(
            model=model,
            init=init,
            n_chains=n_chains,
            n_steps=n_steps,
            id_run=i,
            energy_type=energy_type,
            mlp_models=mlp_models,
            mixture=mixture,
            dim=dim,
            dft_folder_name=dft_folder_name,
            )
        
        mcmc_runs.append(mcmc_run)

        xs.append(mcmc_run["xs"])
        us.append(mcmc_run["us"])
        accs.append(mcmc_run["accs"])
        isomers.append(mcmc_run["isomers"])

        init = join_data(xs[i][-1].clone(), us[i][-1].clone(), isomers[i][-1].clone())

        #TODO: Chech size of xs_for_flows_train

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

        mask_flow = chains_flatten[:, -1]

        dict_new_flows = []

        for mode in modes.detach().numpy().astype(int):

            xs_from_chains = chains_flatten.clone()[mask_flow==mode].clone()

            # extend data set for flow training

            xs_for_flows_train[mode] = torch.cat(
                (xs_for_flows_train[mode].clone(), xs_from_chains.clone())
            )

            #add mlps
            dict_new_flow = train_flow(
                dict_flows_training[i][mode]['model'],
                xs_for_flows_train[mode],
                xs_for_flows_test[mode],
                #mlp_model=get_models(dict_mlps_training[-1])[0],
                **flow_hyperparams[mode],
                dim=dim
                )
            
            dict_new_flows.append(copy.deepcopy(dict_new_flow))
            
        else:
            dict_new_flow = copy.deepcopy(dict_flows_training[i][mode])
            dict_new_flows.append(dict_new_flow)

        dict_flows_training.append(dict_new_flows)

    to_return = {
        "dict_flows_training": dict_flows_training,
        "mcmc_runs": mcmc_runs,
        "xs": xs,
        "us": us,
        "accs": accs,
        "isomers": isomers,
    }

    return to_return