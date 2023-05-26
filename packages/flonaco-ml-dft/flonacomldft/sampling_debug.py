from ase.parallel import parprint as print

import torch
from flonacomldft.utils.io_utils import load_pickle_file, load_csv_file
from flonacomldft.internal_coordinates import Coordinates_mapping
import time

from flonacomldft.utils.io_utils import save_pickle_file
from flonacomldft.utils.io_utils import get_process_id

# Parallel issues with this, add only master process
date_start = time.strftime('%Y-%m-%d %H:%M:%S')
id_process = get_process_id(date_start)

torch.manual_seed(42)

ceph_path = "/mnt/home/amolina/ceph/training_md_converged/is0/flows/"
ana_path = "/home/ana/assisting_sampling/phase_2/histograms_md_training/"

zmat = load_csv_file("database/berendsen/converged/is0_flow_test.csv")
flow_dic = load_pickle_file("is0_flow_dic_training_20230520201411.pkl", path=ceph_path)


coord_mapping = Coordinates_mapping()

xs = coord_mapping.get_real_centered_from_internal(zmat[:, :12], zmat[:, 14], isomer=0, energies=zmat[:, 12])
xs = torch.cat((xs[0], xs[2].reshape(-1, 1), zmat[:, 13].reshape(-1, 1), xs[1].reshape(-1, 1)), dim=1)
xs = xs[torch.randperm(xs.size()[0])] 

n_chains = 50
n_steps = 100

xs_init = xs[:n_chains]

kb = 8.617333262e-5

def run_metropolis(
    model,
    init,
    n_chains,
    n_steps,
    id_run=None,
    energy_type=None,
    frac_dft=0.2,
    mlp_models=None,
    mixture=False,
    T=300,
    with_tqdm=False,
    return_ratio = False,
    return_proposals = False,
    save_schedule = 10,
    id_process = None,
):
    
    beta = 1 / (kb * T)

    from flonacomldft.dft_calculator import DFTCalculator
    
    calculator = DFTCalculator()
    calculator.initialize_calculator()

    x_init = init[:, :12]
    u_init = init[:, 12]
    isomer_init = init[:, 13]
    logdetjac_init = init[:, 14]

    print('init', init)

    xs = []
    us = []
    ratios = []
    accs = []
    xs_prop = []

    for dt in range(n_steps):

        x_new = model.sample(n_chains)
        xs_prop.append(x_new.clone())

        isomer_new = isomer_init.clone()
        print('flow samples', x_new, x_new.shape)

        nll_x = model.nll(x_new)
        nll_x_init = model.nll(x_init)

        print('nll_x', nll_x)
        print('nll_x_init', nll_x_init)

        u_new = torch.zeros(n_chains)

        for i in range(x_new.shape[0]):
            molecule, logdetjac = coord_mapping.build_molecule_from_real_centered(x_new[i].reshape(1, -1), isomer=int(isomer_new[i].item()))
            #from ase.visualize import view
            #view(molecule)
            u_ = calculator.calculate_potential_energy(
                                molecule, 
                                filename='ag6_{:d}_{:d}_{:d}.out'.format(0, dt, i)
                                                )

            print('u_xyz: ', u_)
            u_new[i] = coord_mapping.compute_energy_in_new_frame(u_, logdetjac*(-1))
            print('u: ', u_new[i])

        print('u_init', u_init)
        print('u_new', u_new)

        print('comparison exp: ', beta*(u_init - u_new), -nll_x + nll_x_init)

        ratio = -beta * u_new + nll_x
        ratio += beta * u_init - nll_x_init
        ratio = torch.exp(ratio)    

        ratios.append(ratio.clone())

        print('ratio', ratio)

        rand_num = torch.rand_like(ratio)

        print('rand_num: {}', rand_num)

        prob_acc = torch.min(ratio, torch.ones_like(ratio))

        acc = rand_num < prob_acc                   

        print('acc', acc)

        x_init[acc] = x_new[acc]
        u_init[acc] = u_new[acc]

        xs.append(x_init.clone())
        us.append(u_init.clone())
        accs.append(acc.clone())

        print('acc: {:.2f}'.format(prob_acc.float().mean().item()))

        if dt % save_schedule == 0:
            
            print('checkpointing...')
            
            to_return = {
                'xs': torch.stack(xs),
                'us': torch.stack(us),
                'accs': torch.stack(accs),
                'ratios': torch.stack(ratios),
                'xs_prop': torch.stack(xs_prop)
            }

            from flonacomldft.utils.io_utils import save_pickle_file
            
            save_pickle_file(to_return, 'mcmc_run_{:d}.pkl'.format(id_process))

    to_return = {
        'xs': torch.stack(xs),
        'us': torch.stack(us),
        'accs': torch.stack(accs),
        'ratios': torch.stack(ratios),
        'xs_prop': torch.stack(xs_prop)
    }

    return to_return

mcmc = run_metropolis(flow_dic['model'], 
               xs_init, 
               n_chains, 
               n_steps, 
               id_run=None, 
               energy_type=None, 
               frac_dft=0.2, 
               mlp_models=None, 
               mixture=False, 
               T=300, 
               with_tqdm=False, 
               return_ratio = False, 
               return_proposals = False,
               id_process = id_process)

save_pickle_file(mcmc, 'mcmc_run_{:d}.pkl'.format(id_process))