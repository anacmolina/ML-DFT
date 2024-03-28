"""
Metropolis-Hasting algorithm
"""

import time
import numpy as np
import torch
import tqdm

from flonacomldft.train_mlp_from_data import train_mlp
from flonacomldft.utils.io_utils import save_pickle_file
from ase.parallel import parprint as print
from ase.units import kB


def run_metropolis(model, 
                    init, 
                    n_chains,
                    n_steps,
                    id_run,
                    energy_type,
                    temperature,
                    mixture,
                    mlp_models=None,
                    frac_computed=0.2,
                    dim=12,
                    update_weights=True,
                    scheduler_weights=10,
                    alpha=0.5,
                    return_ratios=True,
                    return_proposals=True,
                    with_tqdm=False,
                    device='cpu',
                    folder_name=None,
                    checkpoints=None,
                    train_mlp_models=False,
                    mlp_init_train=None,
                    mlp_init_test=None,
                    mlp_hyperparams=None,
                    train_mlp_scheduler=None,
                 ):

    """
    Run Metropolis-Hastings algorithm
    Args:
        model: Normalizing Flow Model
        init (torch.Tensor): Initial configurations for the chains
        n_chains (int): Number of chains
        n_steps (int): Number of steps
        id_run (str): Identifier for the run
        energy_type (str): Type of energy to calculate
        temperature (float): Temperature
        mixture (bool): Boolean to indicate if the model is a mixture
        mlp_models (list): List of MLP models
        frac_computed (float): Fraction of configurations to compute
        dim (int): Dimension of the configuration space
        update_weights (bool): Boolean to indicate if the weights of the mixture model are updated
        scheduler_weights (int): Number of steps to update the weights
        alpha (float): Parameter to update weights update
        return_ratios (bool): Boolean to indicate if the ratios are returned
        return_proposals (bool): Boolean to indicate if the proposals are returned
        with_tqdm (bool): Boolean to indicate if tqdm is used
        device (str): Device to use
        folder_name (str): Folder name to save the results
        checkpoints (int): Number of steps to save a checkpoint
        train_mlp_models (bool): Boolean to indicate if the MLP models are trained
        mlp_init_train (list): List of initial configurations to train the MLP models
        mlp_init_test (list): List of initial configurations to test the MLP models
        mlp_hyperparams (dict): Dictionary with hyperparameters for the MLP models
        train_mlp_scheduler (int): Scheduler to train the MLP models
    Returns:
        dict: Dictionary with the results
    """

    assert init.shape[0] == n_chains

    if mixture:
        try:
            assert len(model.models) > 1
        except:
            raise RuntimeError("Model is not a mixture")
    
    print("Running Metropolis-Hastings")

    x_init = init[:, :dim]
    u_init = init[:, dim]
    isomer_init = init[:, dim+1]

    isomer_labels = isomer_init.int().unique().numpy()

    print("Number of isomers: ", isomer_labels)

    beta = 1 / (kB * temperature)

    print("Number of chains: {:d}".format(n_chains))
    print("Number of steps: {:d}".format(n_steps))
    print("Temperature: {:.1f}K".format(temperature))
    print("Energy Type: {:s}".format(energy_type))
    
    if "mlp" in energy_type:
        
        if (mlp_models is None):
            raise RuntimeError("No model to calculate energy")
        
        if mixture:
            print('Mixture model')
            model_mlp_is0, model_mlp_is1 = mlp_models
        else:
            if(isomer_init.sum()==0):
                model_mlp_is0 = mlp_models[0]
            else:
                model_mlp_is1 = mlp_models[0]
        
        print("Use MLP: True")
        print("DFT fraction: {:.1f}".format(frac_computed))

        if train_mlp_models:

            idx_train = 0

            xs_for_mlps_train = [ mlp_init_train[i] for i in isomer_labels ]
            xs_for_mlps_test = [ mlp_init_test[i] for i in isomer_labels ]

            dict_mlp_models = []

    else:
        
        print("No MLP")

    if ("dft" in energy_type) or ("emt" in energy_type):

        use_calc = True

        xs_calc = []
        us_calc = []
        isomers_calc = []
        inds_calc = []

        if "emt" in energy_type:

            from flonacomldft.dft_calculator import EMTCalculator
            from flonacomldft.internal_coordinates import Coordinates_mapping

            coord_mapping = Coordinates_mapping(etype='emt')
            calculator = EMTCalculator()

            print("Use EMT Calculator: {:s}".format(str(use_calc)))

        if "dft" in energy_type:

            import gpaw.mpi as mpi
            from flonacomldft.dft_calculator import DFTCalculator
            from flonacomldft.internal_coordinates import Coordinates_mapping
        
            coord_mapping = Coordinates_mapping(etype='dft')
            calculator = DFTCalculator()

            mpi.world.barrier()

            if folder_name is not None:
                calculator.initialize_calculator(foldername=folder_name)
            else:
                calculator.initialize_calculator()

            print("Use DFT Calculator: {:s}".format(str(use_calc)))

    else:
    
        use_calc = False

    if return_ratios:
        ratios = []

    if return_proposals:
        xs_proposals = []
        us_proposals = []
        isomers_proposals = []
        nlls_proposals = []

    print("Mixture Model: {:s}".format(str(mixture)))

    if mixture:
        weights = [model.weights.clone().detach()]

    xs = []
    us = []
    accs = []
    isomers = []
    nlls = []

    time_step_mcmc = []

    if with_tqdm == False:

        print("Step \t Acc Rate")

    if with_tqdm:
        pbar = tqdm.tqdm(range(n_steps))
    else:
        pbar = range(n_steps)

    def write_not_compute(id_run, dt, i):
        """
        Write file with not computed molecules
        Args:
            id_run (str): Identifier for the run
            dt (int): Step
            i (int): Chain index
        """

        with open('not_computed_molecules.txt', 'w') as f:

            f.write("Molecule run {:s} step {:d} chain {:d} not computed\n".format(
                str(id_run),
                dt, 
                i))

    for dt in pbar:

        if mixture:
            x_new, isomer_new = model.sample(n_chains, return_mus=True)
        else:
            x_new = model.sample(n_chains)
            isomer_new = isomer_init.clone()

        x_new = x_new.clone().detach() #set float
        isomer_new = isomer_new.clone().detach() #set dype
        
        nll_x = model.nll(x_new)
        nll_x_init = model.nll(x_init)

        if return_proposals:
            xs_proposals.append(x_new.clone().detach())
            isomers_proposals.append(isomer_new.clone().detach())
            nlls_proposals.append(nll_x.clone().detach())

        if "mlp" in energy_type:

            u_new = torch.zeros((n_chains, 1)).squeeze()

            if mixture:

                u_new[~isomer_new.bool()] = model_mlp_is0(x_new[~isomer_new.bool()]).reshape(1, -1)
                u_new[isomer_new.bool()] = model_mlp_is1(x_new[isomer_new.bool()]).reshape(1, -1)

            else:

                if(isomer_new.sum()==0):
                    u_new = model_mlp_is0(x_new)
                else:
                    u_new = model_mlp_is1(x_new)

            u_new = u_new.squeeze()

        if use_calc:

            ind_not_computed = torch.zeros(n_chains)
            ind_computed = torch.zeros(n_chains)

            if (energy_type == "dft") or (energy_type == "emt"):

                ind_computed = torch.ones(n_chains)
                u_new = torch.zeros((n_chains))

            else:

                n_computed = int(u_new.shape[0] * frac_computed)
                u_sort, ind_u_sort = u_new.sort()
                
                for idx in ind_u_sort[:n_computed]:
                    
                    ind_computed[idx] = 1

            for i, flag_computed in enumerate(ind_computed):

                if flag_computed:

                    try: 

                        molecule, logdetjac = coord_mapping.build_molecule_from_real_centered(
                            x_new[i].reshape(1, -1), 
                            isomer=isomer_new[i].int().item(),
                        )

                        input_calculator = {'atoms': molecule, }

                        if "dft" in energy_type:
                        
                            input_calculator['filename'] = 'ag6_{:s}_{:d}_{:d}.out'.format(
                                str(id_run), dt, i
                            )

                            mpi.world.barrier()

                        u = calculator.calculate_potential_energy(**input_calculator)

                        u_new[i] = coord_mapping.compute_energy_in_new_frame(
                            u,
                            logdetjac*(-1),
                            temperature=temperature,
                        )

                        xs_calc.append(x_new[i].clone().detach())
                        us_calc.append(u_new[i].clone().detach())
                        isomers_calc.append(isomer_new[i].clone().detach())

                    except:

                        print("Molecule {:d} not computed".format(i))

                        ind_not_computed[i] = 1
                        u_new[i] = 0.0

                        if "dft" in energy_type:
                            
                            rank = mpi.world.rank
                            
                            if rank==0:
                                write_not_compute(x_new[i], isomer_new[i], id_run, dt, i)
                        else:
                            
                            write_not_compute(x_new[i], isomer_new[i], id_run, dt, i)

        if use_calc and "dft" in energy_type:

            time_step_mcmc.append(time.time())

            mpi.world.barrier()

        else:

            time_step_mcmc.append(time.time())

        if return_proposals:
            us_proposals.append(u_new.clone().detach())

        ratio = -beta * u_new + nll_x
        ratio += beta * u_init - nll_x_init
        ratio = torch.exp(ratio)

        if return_ratios:
            ratios.append( torch.min(ratio.clone().detach(), 
                            torch.ones_like(ratio)) )

        s = torch.rand_like(ratio)

        acc = s < torch.min(ratio.detach(), torch.ones_like(ratio))

        if use_calc:
    
            if ind_not_computed.sum() > 0:

                acc[ind_not_computed.bool()] = False

            inds_calc.append(ind_computed)

        x_new[~acc] = x_init[~acc]
        u_new[~acc] = u_init[~acc]

        if mixture:
        
            isomer_new[~acc] = isomer_init[~acc]
        
        else:

            isomer_new = isomer_init.clone()
        
        xs.append(x_new.clone().detach())
        us.append(u_new.clone().detach())
        accs.append(acc.clone().detach())
        isomers.append(isomer_new.clone().detach())
        nlls.append(model.nll(x_new).clone().detach())

        x_init = x_new.clone().detach()
        u_init = u_new.clone().detach()
        isomer_init = isomer_new.clone().detach()

        if with_tqdm:
            pbar.set_description("Step {:d} \t Acceptance Rate {:.3f}".format(
                dt, acc.float().mean().item()))
        else:

            if "dft" in energy_type:
                
                mpi.world.barrier()
                
                print("{:d} \t {:.3f}".format( dt, 
                                        acc.float().mean().item() 
                                        )
                )
            else:
                print("{:d} \t {:.3f}".format( dt, 
                                        acc.float().mean().item()
                                        )
            )

        if mixture and update_weights and dt > 0 and dt % scheduler_weights == 0:

            window_weights = torch.stack( [(~torch.stack(isomers).bool())[-scheduler_weights:].float().mean(),
                              torch.stack(isomers)[-scheduler_weights:].float().mean()] )

            new_weights = alpha * model.weights.clone() + (1 - alpha) * window_weights.clone()

            if torch.all(new_weights < 0.75):

                model.weights = new_weights.clone()

            weights.append(model.weights.clone().detach())

        if checkpoints is not None and dt > 0 and dt % checkpoints == 0:

            print("Saving checkpoint {:d}".format(dt))

            checkpoint_data = {
                'xs': torch.stack(xs).clone(),
                'us': torch.stack(us).clone(),
                'accs': torch.stack(accs).clone(),
                'isomers': torch.stack(isomers).clone(),
                'nlls': torch.stack(nlls).clone(),
                'time_mcmc': time_step_mcmc,
            }

            if return_ratios:
                checkpoint_data['ratios'] = torch.stack(ratios).clone()

            if return_proposals:
                checkpoint_data['xs_proposals'] = torch.stack(xs_proposals).clone()
                checkpoint_data['us_proposals'] = torch.stack(us_proposals).clone()
                checkpoint_data['isomers_proposals'] = torch.stack(isomers_proposals).clone()
                checkpoint_data['nlls_proposals'] = torch.stack(nlls_proposals).clone()

            if use_calc:
                    
                checkpoint_data['xs_calc'] = torch.stack(xs_calc).clone()
                checkpoint_data['us_calc'] = torch.stack(us_calc).clone()
                checkpoint_data['isomers_calc'] = torch.stack(isomers_calc).clone()
                checkpoint_data['inds_calc'] = torch.stack(inds_calc).clone()

            if mixture and update_weights:

                checkpoint_data['weights'] = torch.stack(weights).clone()

            save_pickle_file(
                checkpoint_data,
                'checkpoint_{:s}_{:d}.pkl'.format(str(id_run), dt),
                path=folder_name.split('/')[-2],
            )

        if train_mlp_models and use_calc and dt > 0 and dt % train_mlp_scheduler == 0:

            new_mlp_dataset = torch.cat(
                  (torch.stack(xs_calc).clone(),
                  torch.stack(us_calc).reshape(-1, 1).clone(),
                  torch.stack(isomers_calc).reshape(-1, 1).clone()),
                  dim=1)[idx_train:]
            
            print('new_mlp_dataset shape: ', new_mlp_dataset.shape)
            
            mask_mlp = new_mlp_dataset[:, -1]

            for mode in isomer_labels:

                print("Training MLP for isomer {:d}".format(mode))

                mask_mlp_mode = mask_mlp == isomer_labels[mode]
                xs_chains_dataset = new_mlp_dataset[mask_mlp_mode]

                print('Shape xs: ', xs_chains_dataset.shape)

                indexes = torch.randperm(xs_chains_dataset.shape[0])

                split_ratio = 0.8

                split_index = int(xs_chains_dataset.shape[0] * split_ratio)
                
                xs_chains_dataset_train = xs_chains_dataset[indexes[:split_index]]
                xs_chains_dataset_test = xs_chains_dataset[indexes[split_index:]]

                print('MLP chains train dataset shape: ', list(xs_chains_dataset_train.shape))
                print('MLP chains test dataset shape: ', list(xs_chains_dataset_test.shape))

                xs_for_mlps_train[mode] = torch.cat(
                        (xs_for_mlps_train[mode].clone(), 
                         xs_chains_dataset_train.clone())
                    )
                
                xs_for_mlps_test[mode] = torch.cat(
                        (xs_for_mlps_test[mode].clone(), 
                         xs_chains_dataset_test.clone())
                    )

                print('xs_for_mlps_train shape: ', xs_for_mlps_train[mode].shape)
                print('xs_for_mlps_test shape: ', xs_for_mlps_test[mode].shape)      

                dict_new_mlp = train_mlp(
                    model=mlp_models[mode],
                    train=xs_for_mlps_train[mode],
                    test=xs_for_mlps_test[mode],
                    **mlp_hyperparams[mode],
                    dim=dim,
                )

                dict_mlp_models.append(dict_new_mlp)

            idx_train = dt

            print('new idx_train: ', idx_train)

            mlp_models = [dict_mlp_models[mode]['model'] for mode in isomer_labels]

            if mixture:
                model_mlp_is0, model_mlp_is1 = mlp_models
            else:
                if(isomer_init.sum()==0):
                    model_mlp_is0 = mlp_models[0]
                else:
                    model_mlp_is1 = mlp_models[0]

    to_return = {
        'xs': torch.stack(xs),
        'us': torch.stack(us),
        'accs': torch.stack(accs),
        'isomers': torch.stack(isomers),
        'nlls': torch.stack(nlls),
        'time_mcmc': time_step_mcmc,
    }

    if return_ratios:
        to_return['ratios'] = torch.stack(ratios)

    if return_proposals:
        to_return['xs_proposals'] = torch.stack(xs_proposals)
        to_return['us_proposals'] = torch.stack(us_proposals)
        to_return['isomers_proposals'] = torch.stack(isomers_proposals)
        to_return['nlls_proposals'] = torch.stack(nlls_proposals)

    if use_calc:
            
        to_return['xs_calc'] = torch.stack(xs_calc)
        to_return['us_calc'] = torch.stack(us_calc)
        to_return['isomers_calc'] = torch.stack(isomers_calc)
        to_return['inds_calc'] = torch.stack(inds_calc)

    if mixture and update_weights:
        to_return['weights'] = torch.stack(weights)

    if train_mlp_models:
        to_return['dict_mlps'] = dict_mlp_models
        to_return['mlps_datasets'] = {'train': xs_for_mlps_train, 
                                    'test': xs_for_mlps_test}

    return to_return

