import time
import copy
import torch
from abflowmc.train_flow_from_data import train_flow
from abflowmc.train_mlp_from_data import train_mlp
from abflowmc.sampling import run_metropolis
from abflowmc.models.mixture import Mixture

from ase.parallel import parprint as print

def Transpose(x):
    """
    Transpose the input tensor x
    Args:
        x: input tensor
    Returns:
        The transposed tensor
    """
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

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
    scheduler_train_mlp_models=5,
    frac_computed=0.3,
    update_frac_computed=True,
    scheduler_frac_computed=5,
    min_frac_computed=0.3,
    init_weights=None, 
    update_weights=True,
    scheduler_weights=10,
    alpha=0.5,
    n_samples_train_flow=None,
    folder_name=None,
    device='cpu', 
    ):

    """
    Adaptive sampling function
    Args:
        mcmc_init (torch.Tensor): the initial configurations
        n_chains (int): the number of chains
        n_steps (int): the number of steps
        n_runs (int): the number of runs
        flow_init_train (list): the initial training data for the flows
        dict_flows_init (list): the initial flow models
        flow_hyperparams (list): the hyperparameters for the flows
        energy_type (str): the type of energy model
        temperature (float): the temperature
        mixture (bool): whether to use a mixture model
        dim (int): the dimension of the input data
        dict_mlps_init (list): the initial MLP models
        mlp_init_train (list): the initial training data for the MLPs
        mlp_init_test (list): the initial test data for the MLPs
        mlp_hyperparams (list): the hyperparameters for the MLPs
        train_mlp_models (bool): whether to train the MLP models
        scheduler_train_mlp_models (int): the scheduler for training the MLP models 
        frac_computed (float): the fraction of computed energies
        update_frac_computed (bool): whether to update the fraction of computed energies
        scheduler_frac_computed (int): the scheduler for updating the fraction of computed energies
        min_frac_computed (float): the minimum fraction of computed energies
        init_weights (torch.Tensor): the initial weights for the mixture model
        update_weights (bool): whether to update the weights
        scheduler_weights (int): the scheduler for updating the weights
        alpha (float): the alpha parameter for the mixture model
        n_samples_train_flow (torch.Tensor): the number of samples for training the flows
        folder_name (str): the folder name
        device (str): the device
    Returns:
        dict: a dictionary containing
            xs (list): the configurations
            us (list): the energies
            accs (list): the acceptance rates
            isomers (list): the isomers
            nlls (list): the negative log-likelihoods
            time_mcmc (list): the time steps
            time_step_flow (list): the time steps for the flows
            time_step_adaptive (list): the time steps for the adaptive sampling
            xs_proposals (list): the proposals
            us_proposals (list): the energies of the proposals
            isomers_proposals (list): the isomers of the proposals
            nlls_proposals (list): the negative log-likelihoods of the proposals
            dict_flows (list): the flow models
            flows_dataset (list): the training data for the flows
            xs_calc (list): the configurations from the DFT calculations
            us_calc (list): the energies from the DFT calculations
            isomers_calc (list): the isomers from the DFT calculations
            inds_calc (list): the indexes from the DFT calculations
            dict_mlps (list): the MLP models
            mlps_datasets (dict): the training and test data for the MLPs
    """

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

            mlp_models = [dict_mlp['model'] for dict_mlp in dict_mlps[-1]]

        if update_frac_computed and (i % scheduler_frac_computed == 0) and (i > 0):
            print('Updating fraction of computed energies')
            if frac_computed > min_frac_computed:
                frac_computed = frac_computed / 2
                if frac_computed < min_frac_computed:
                    frac_computed = min_frac_computed
            else:
                frac_computed = min_frac_computed
            print('New fraction of computed energies: ', frac_computed)

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

            print('DFT computations: ', xs_calc_run.shape, us_calc_run.shape, isomers_calc_run.shape)

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

                if (i % scheduler_train_mlp_models == 0) and (i > 0):
                      
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

            if (i % scheduler_train_mlp_models == 0) and (i > 0):

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
