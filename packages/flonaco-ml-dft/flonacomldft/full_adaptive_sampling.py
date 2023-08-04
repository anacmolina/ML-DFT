import torch
import time
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
        path=None,
        fix_train_samples=None,
        save_ratios = 5,
        ):

    modes = torch.unique(torch.cat(flow_init_train)[:, dim+1]).int()

    print('modes: ', modes)


    xs_for_flows_train = [ flow_init_train[i] for i in range(len(flow_init_train)) ]
    xs_for_flows_test = [ flow_init_test[i] for i in range(len(flow_init_test)) ]

    dict_flows_training = [copy.deepcopy(dict_flows_init), ]

    if 'mlp' in energy_type or dict_mlps_init is not None:
        mlp_models = get_models(dict_mlps_init)
    else:
        mlp_models = [None]*len(modes)

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

    if fix_train_samples is None:
        fix_train_samples = xs_for_flows_train[0].shape[0]
    elif fix_train_samples > xs_for_flows_train[0].shape[0]:
        fix_train_samples = xs_for_flows_train[0].shape[0]

    timestep_flow = []
    timestep_adaptive = []    

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
            dft_folder_name=path + '/DFTAdaptive',
            )
        
        timestep_adaptive.append(time.time())

        mcmc_runs.append(mcmc_run)

        xs.append(mcmc_run["xs"])
        us.append(mcmc_run["us"])
        accs.append(mcmc_run["accs"])
        isomers.append(mcmc_run["isomers"])

        init = join_data(xs[i][-1].clone(), us[i][-1].clone(), isomers[i][-1].clone())

        #TODO: Chech size of xs_for_flows_train, if rewriting first and last element in the list

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

        print("modes: ", modes)

        for mode in range(len(modes)):#modes.detach().numpy().astype(int):
            print("mode: ", mode)

            xs_from_chains = chains_flatten.clone()[mask_flow==modes[mode]].clone()

            # extend data set for flow training

            print("shape xs_for_flows_train: ", xs_for_flows_train[mode].shape)
            print("shape xs_from_chains: ", xs_from_chains.shape)

            xs_for_flows_train[mode] = torch.cat(
                (xs_for_flows_train[mode].clone(), xs_from_chains.clone())
            )

            # n_runs/save_splits
            if i % save_ratios == 0 and flow_hyperparams[mode]['compute_part_ratio']:
                #TODO: do this only for some number of runs
                #flow_hyperparams[mode]['compute_part_ratio'] = True
                if "dft" in energy_type:
                    flow_hyperparams[mode]['path'] = path + '/DFTRatios_{:d}'.format(i)
                else:
                    flow_hyperparams[mode]['path'] = ''            
            #else:
            #    flow_hyperparams[mode]['compute_part_ratio'] = False

            print("shape xs_for_flows_train: ", xs_for_flows_train[mode][:].shape)

            model = copy.deepcopy(dict_flows_training[i][mode]['model'])

            #add mlps
            dict_new_flow = train_flow(
                model,
                xs_for_flows_train[mode][-fix_train_samples:],
                xs_for_flows_test[mode],
                #mlp_model=get_models(dict_mlps_training[-1])[0],
                **flow_hyperparams[mode],
                dim=dim,
                mlp_model=mlp_models[mode],
                )
                    
            timestep_flow.append(time.time())
            
            #if (i+1)*n_steps >= 100:
            #    flow_hyperparams[mode]['lr'] = 1e-4
            #    flow_hyperparams[mode]['n_iter'] = 10
            
            dict_new_flows.append(dict_new_flow)
            
        else:
            dict_new_flow = dict_flows_training[i][mode]
            dict_new_flows.append(dict_new_flow)

        dict_flows_training.append(dict_new_flows)

    to_return = {
        "dict_flows_training": dict_flows_training,
        "mcmc_runs": mcmc_runs,
        "xs": xs,
        "us": us,
        "accs": accs,
        "isomers": isomers,
        "time_flow": timestep_flow,
        "time_adaptive": timestep_adaptive,
    }

    return to_return