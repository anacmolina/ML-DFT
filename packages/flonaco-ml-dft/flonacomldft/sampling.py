"""
Script with Metropolis-Hasting algorithm. 

"""

import time
import numpy as np
import torch
import tqdm

from flonacomldft.collective_variables import get_collective_variables
from ase.parallel import parprint as print
from ase.units import kB

#TODO: Add calculator for MLP
#TODO: Add cv values to return
#TODO: add docstrings

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
                 ):

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

    beta = 1 / (kB * temperature)

    print("Number of chains: {:d}".format(n_chains))
    print("Number of steps: {:d}".format(n_steps))
    print("Temperature: {:.1f}K".format(temperature))
    print("Energy Type: {:s}".format(energy_type))

    if "mlp" in energy_type:
        #TODO: Add calculator for MLP
        
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
        
        print("Use Neural Predictor: True")
    
    else:
        
        print("No Neural Predictor")

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
        xs_proposals = [init[:, :dim]]
        us_proposals = [init[:, dim]]
        isomers_proposals = [init[:, dim+1]]
        nlls_proposals = [model.nll(init[:, :dim])]

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

    def write_not_compute(x, isomer, id_run, dt, i):

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
            xs_proposals.append(x_new)
            isomers_proposals.append(isomer_new)
            nlls_proposals.append(nll_x)

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

                    #try: 

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

                    #u_new[i] = model_mlp_is0(x_new[i].reshape(1, -1))

                    xs_calc.append(x_new[i])
                    us_calc.append(u_new[i])
                    isomers_calc.append(isomer_new[i])

                    #except:
#
                    #    print("Molecule {:d} not computed".format(i))
#
                    #    ind_not_computed[i] = 1
                    #    u_new[i] = 0.0
#
                    #    if "dft" in energy_type:
                    #        
                    #        rank = mpi.world.rank
                    #        
                    #        if rank==0:
                    #            write_not_compute(x_new[i], isomer_new[i], id_run, dt, i)
                    #    else:
                    #        
                    #        write_not_compute(x_new[i], isomer_new[i], id_run, dt, i)

        if use_calc and "dft" in energy_type:

            rank = mpi.world.rank

            if rank == 0:

                time_step_mcmc.append(time.time())

            mpi.world.barrier()

        else:

            time_step_mcmc.append(time.time())

        if return_proposals:
            us_proposals.append(u_new)

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

    return to_return

