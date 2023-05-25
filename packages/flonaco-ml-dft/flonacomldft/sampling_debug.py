import os
import torch
from flonacomldft.utils.io_utils import load_pickle_file, load_csv_file
from flonacomldft.internal_coordinates import Coordinates_mapping

zmat = load_csv_file("database/berendsen/converged/is0_flow_train.csv")
flow_dic = load_pickle_file("home/ana/assisting_sampling/phase_2/histograms_md_training/is0_flow_dic_training_20230520201411.pkl", path="/")

coord_mapping = Coordinates_mapping()

xs = coord_mapping.get_real_centered_from_internal(zmat[:, :12], zmat[:, 14], isomer=0, energies=zmat[:, 12])
xs = torch.cat((xs[0], xs[2].reshape(-1, 1), zmat[:, 13].reshape(-1, 1), xs[1].reshape(-1, 1)), dim=1)
xs = xs[torch.randperm(xs.size()[0])] 

n_chains = 3

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

    x_new = model.sample(n_chains)
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
                            filename='ag6_{:d}_{:d}_{:d}.out'.format(0, 0, i)
                                            )

        print('u_xyz: ', u_)
        u_new[i] = coord_mapping.compute_energy_in_new_frame(u_, logdetjac*(-1))
        print('u: ', u_new[i])

    print('u_init', u_init)
    print('u_new', u_new)


    ratio = -beta * u_new + nll_x
    ratio += beta * u_init - nll_x_init
    ratio = torch.exp(ratio)    

    print('ratio', ratio)

    rand_num = torch.rand_like(ratio)

    acc = rand_num < torch.min(ratio, torch.ones_like(ratio))                    

    print('acc', acc)

    return acc, ratio

print(run_metropolis(flow_dic['model'], 
               xs_init, 
               n_chains, 
               5, 
               id_run=None, 
               energy_type=None, 
               frac_dft=0.2, 
               mlp_models=None, 
               mixture=False, 
               T=300, 
               with_tqdm=False, 
               return_ratio = False, 
               return_proposals = False))