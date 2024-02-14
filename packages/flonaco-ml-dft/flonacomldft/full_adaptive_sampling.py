import time
import copy
import torch
from flonacomldft.train_flow_from_data import train_flow
from flonacomldft.train_mlp_from_data import train_mlp
from flonacomldft.sampling import run_metropolis
from flonacomldft.models.mixture import Mixture

from ase.parallel import parprint as print

def Transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

#TODO: add docstrings

def run_adaptive_sampling(
    mcmc_init,
    n_chains,
    n_steps,
    n_runs,
    flow_init_train,
    dict_flows_init,
    flow_hyperparams,
    energy_type,
    temperature,
    mixture,
    dim=12,
    dict_mlps_init=None,
    mlp_init_train=None,
    mlp_init_test=None,
    mlp_hyperparams=None,
    train_mlp_models=True,
    frac_computed=0.2,
    init_weights=None, 
    update_weights=True,
    scheduler_weights=10,
    alpha=0.5,
    n_samples_train_flow=None,
    folder_name=None,
    device='cpu', 
    ):

    print('Adaptive sampling')
    print('Number of runs: ', n_runs)
    print('Number of chains: ', n_chains)
    print('Number of steps: ', n_steps)
    print('Temperature: ', temperature)
    print('Energy type: ', energy_type)
    print('Mixture model: ', mixture)

    isomer_labels = torch.unique(torch.cat(flow_init_train)[:, dim+1]).int()
    n_isomers = isomer_labels.shape[0]

    assert mixture == (n_isomers > 1), 'Model setup is not consistent with the number of isomers'

    print('Isomer labels: ', isomer_labels.tolist())

    xs_for_flows_train = [ flow_init_train[i] for i in range(n_isomers) ]

    print('Flow train dataset shapes: ', [ list(xs_for_flows_train[i].shape) for i in range(n_isomers) ])

    if (energy_type == 'mlp') or (energy_type == 'dft') or (energy_type == 'emt'):
        train_mlp_models = False

    if ('mlp' in energy_type) and (dict_mlps_init is not None):

        dict_mlps = [ copy.deepcopy(dict_mlps_init) ]

        if train_mlp_models:

            xs_for_mlps_train = [ mlp_init_train[i] for i in range(n_isomers) ]
            xs_for_mlps_test = [ mlp_init_test[i] for i in range(n_isomers) ]

            #n_samples_train_mlp = torch.tensor( [ xs_for_mlps_train[i].shape[0] for i in range(n_isomers) ] )
            #mlp_batch_size = torch.tensor( [ mlp_hyperparams[i]['bs'] for i in range(n_isomers) ] )
#
            #fix_iters_per_batch = torch.ceil(n_samples_train_mlp / mlp_batch_size).clone().detach().int()


            print('MLP train dataset shapes: ', [ list(xs_for_mlps_train[i].shape) for i in range(n_isomers) ])
            print('MLP test dataset shapes: ', [ list(xs_for_mlps_test[i].shape) for i in range(n_isomers) ])

        else:

            if ('mlp' in energy_type) and (train_mlp_models == False):
            
                mlp_models = [dict_mlp['model'] for dict_mlp in dict_mlps[-1]]

            print('MLPs will not be trained')

    else:

        mlp_models = [None] * n_isomers

        print('MLPs will not be used')

    if mixture and init_weights is None:
        
        init_weights = torch.tensor([1/n_isomers for i in range(n_isomers)]).detach()
        print('Initial weights: ', init_weights.tolist())
    
    elif mixture and init_weights is not None:
        
        print('Initial weights: ', init_weights.tolist())

    if n_samples_train_flow is None:
        n_samples_train_flow = torch.tensor( [ n_chains * n_steps * 5  +  xs_for_flows_train[i].shape[0] for i in range(n_isomers) ] )
    
    print('Number of samples for training flows: ', n_samples_train_flow)
    
    if ("dft" in energy_type) or ("emt" in energy_type):

        use_calc = True

        xs_calc = []
        us_calc = []
        isomers_calc = []
        inds_calc = []

    else:

        use_calc = False

    dict_flows = [ copy.deepcopy(dict_flows_init) ]

    xs_proposals = []
    us_proposals = []
    isomers_proposals = []
    nlls_proposals = []

    xs = []
    us = []
    accs = []
    isomers = []
    nlls = []
    time_mcmc = []

    time_step_flow = []
    time_step_adaptive = []

    init = mcmc_init

    for i in range(n_runs):

        print('Run: ', i)

        if mixture:
            flow_models = [ dict_flows[i][j]['model'] for j in range(n_isomers)]
            model = Mixture(flow_models, init_weights)
            print('Current weights: ', model.weights.tolist())
        else:
            model = dict_flows[i][0]['model']

        if ('mlp' in energy_type) and (train_mlp_models == True):

            mlp_models = [dict_mlp['model'] for dict_mlp in dict_mlps[i]]

        mcmc = run_metropolis(model = model, 
                                init = init, 
                                n_chains = n_chains,
                                n_steps = n_steps,
                                id_run = i, 
                                energy_type = energy_type, 
                                temperature = temperature,
                                mixture = mixture,
                                mlp_models = mlp_models,
                                frac_computed = frac_computed,
                                dim = dim,
                                update_weights = update_weights,
                                scheduler_weights = scheduler_weights,
                                alpha = alpha,
                                return_ratios = False,
                                return_proposals = True,
                                with_tqdm = False,
                                folder_name = folder_name + '/DFTAdaptive',
                                device=device,
                                )
        
        time_step_adaptive.append(time.time())

        xs.append(mcmc['xs'])
        us.append(mcmc['us'])
        accs.append(mcmc['accs'])
        isomers.append(mcmc['isomers'])
        nlls.append(mcmc['nlls'])
        time_mcmc.append(mcmc['time_mcmc'])

        xs_proposals.append(mcmc['xs_proposals'])
        us_proposals.append(mcmc['us_proposals'])
        isomers_proposals.append(mcmc['isomers_proposals'])
        nlls_proposals.append(mcmc['nlls_proposals'])

        if use_calc:

            xs_calc.append(mcmc['xs_calc'])
            us_calc.append(mcmc['us_calc'])
            isomers_calc.append(mcmc['isomers_calc'])
            inds_calc.append(mcmc['inds_calc'])

        init = torch.cat( (xs[i][-1].clone(),
            us[i][-1].clone().reshape(-1, 1),                 
            isomers[i][-1].clone().reshape(-1, 1)), 
            dim=1)
        
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


        chains_flatten = chains_flatten.reshape(n_steps * n_chains, 
                                                chains_flatten.shape[-1])
        
        mask_flow = chains_flatten[:, -1]

        dict_new_flows = []

        if train_mlp_models and use_calc:

            dict_new_mlps = []

            xs_calc_run = mcmc['xs_calc']
            us_calc_run = mcmc['us_calc']
            isomers_calc_run = mcmc['isomers_calc']

            configs_dft_flatten = Transpose(
                torch.cat(
                        (
                            Transpose(xs_calc_run.clone()),
                            us_calc_run.clone().reshape(1, -1),
                            isomers_calc_run.reshape(1, -1),
                        ),
                        dim=0,
                    )
                )

            mask_mlp = configs_dft_flatten[:, -1]

        for mode in range(n_isomers):

            print('Isomer: ', isomer_labels[mode].item())

            xs_from_chains = chains_flatten.clone()[mask_flow == isomer_labels[mode]].clone()

            print('Flow train dataset shape: ', list(xs_for_flows_train[mode].shape))
            print('Flow chains dataset shape: ', list(xs_from_chains.shape))

            xs_for_flows_train[mode] = torch.cat(
                    (xs_for_flows_train[mode].clone(), xs_from_chains.clone())
                )
            
            print('Flow train dataset shape: ', list(xs_for_flows_train[mode].shape))

            print('Flow input train dataset shape: ', list(xs_for_flows_train[mode][-n_samples_train_flow[mode]:].shape) )
        
            flow_model = copy.deepcopy(dict_flows[i][mode]['model'])

            dict_new_flow = train_flow(
                model=flow_model,
                train=xs_for_flows_train[mode][-n_samples_train_flow[mode]:],
                **flow_hyperparams[mode],
                dim=dim,
            )

            time_step_flow.append(time.time())

            dict_new_flows.append(dict_new_flow)

            if train_mlp_models and use_calc:

                mask_mlp_mode = mask_mlp == isomer_labels[mode]

                indexes = torch.randperm(configs_dft_flatten[mask_mlp_mode].shape[0])

                split_ratio = 0.8
                split_index = int(configs_dft_flatten[mask_mlp_mode].shape[0] * split_ratio)
                configs_dft_flatten_train = configs_dft_flatten[mask_mlp_mode][indexes[:split_index]]
                configs_dft_flatten_test = configs_dft_flatten[mask_mlp_mode][indexes[split_index:]]

                print('MLP chains train dataset shape: ', list(configs_dft_flatten_train.shape))
                print('MLP chains test dataset shape: ', list(configs_dft_flatten_test.shape))

                xs_for_mlps_train[mode] = torch.cat(
                        (xs_for_mlps_train[mode].clone(), 
                         configs_dft_flatten_train.clone())
                    )
                
                xs_for_mlps_test[mode] = torch.cat(
                        (xs_for_mlps_test[mode].clone(), 
                         configs_dft_flatten_test.clone())
                    )
                
                print('MLP train dataset shape: ', list(xs_for_mlps_train[mode].shape))
                print('MLP test dataset shape: ', list(xs_for_mlps_test[mode].shape))

                #new_batch_size = torch.ceil(xs_for_mlps_train[mode].shape[0] / fix_iters_per_batch[mode]).int().item()
                #mlp_hyperparams[mode]['bs'] = new_batch_size

                #print("Number of gradient steps per epoch: ", new_batch_size, 
                #      #fix_iters_per_batch[mode],
                #      #xs_for_mlps_train[mode].shape[0],
                #      int(xs_for_mlps_train[mode].shape[0] / new_batch_size)*mlp_hyperparams[mode]['n_iter'] )
                      
                dict_new_mlp = train_mlp(
                    model=mlp_models[mode],
                    train=xs_for_mlps_train[mode],
                    test=xs_for_mlps_test[mode],
                    **mlp_hyperparams[mode],
                    dim=dim,
                )

                dict_new_mlps.append(dict_new_mlp)

        dict_flows.append(dict_new_flows)

        if train_mlp_models and use_calc:

            dict_mlps.append(dict_new_mlps)

        if mixture and update_weights:
        
            init_weights = model.weights

    to_return = {
        'xs': xs,
        'us': us,
        'accs': accs,
        'isomers': isomers,
        'nlls': nlls,
        'time_mcmc': time_mcmc,
        'time_step_flow': time_step_flow,
        'time_step_adaptive': time_step_adaptive,
        'xs_proposals': xs_proposals,
        'us_proposals': us_proposals,
        'isomers_proposals': isomers_proposals,
        'nlls_proposals': nlls_proposals,
        'dict_flows': dict_flows,
        'flows_dataset': xs_for_flows_train,
    }

    if use_calc:
            
        to_return['xs_calc'] = xs_calc
        to_return['us_calc'] = us_calc
        to_return['isomers_calc'] = isomers_calc
        to_return['inds_calc'] = inds_calc

    if ('mlp' in energy_type):
        to_return['dict_mlps'] = dict_mlps

    if train_mlp_models:
    
        to_return['mlps_datasets'] =  {'train': xs_for_mlps_train,
                        'test': xs_for_mlps_test}

    return to_return
