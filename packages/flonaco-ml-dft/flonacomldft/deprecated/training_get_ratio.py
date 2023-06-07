import torch
from ase.units import kB


#kb = 8.617333262e-5

def compute_ratio(u, u_init, nll, nll_init, beta):
    return torch.exp(-beta * (u - u_init) + nll - nll_init)

def compute_pariticpation_ratio(x_new, u_new, nll, beta):
    log_weight = (- u_new * beta).squeeze() + nll.squeeze()
    log_ratio = torch.logsumexp(2 * log_weight, dim=0) - 2 * torch.logsumexp(log_weight, dim=0) 
    return torch.exp(-log_ratio) / x_new.shape[0]

def get_all_ratios(
    model,
    init,
    n_chains,
    n_steps,
    mlp_model=None,
    scheduled_dft=100,
    T=300,):

    assert init.shape[0] == n_chains

    x_init = init[:, :12]
    u_init = init[:, 12]
    isomer_init = init[:, 13]

    beta = 1 / (kB * T)

    # internal coordinates transformations
    from flonacomldft.internal_coordinates import Coordinates_mapping
    coord_mapping = Coordinates_mapping()

    # dft calculator
    from flonacomldft.dft_calculator import DFTCalculator
    calculator = DFTCalculator()
    calculator.initialize_calculator()

    mlp_ratios = []
    dft_ratios = []
    mlp_part_ratios = []
    dft_part_ratios = []

    for dt in range(n_steps):
        x_new = model.sample(n_chains)
        isomer_new = isomer_init

        nll_x = model.nll(x_new)
        nll_x_init = model.nll(x_init)

        # calculate energy mlp
        if mlp_model is not None:
            
            u_new_mlp = mlp_model(x_new)
            ratio_mlp = compute_ratio(u_new_mlp, u_init, nll_x, nll_x_init, beta)
            participation_ratio_mlp = compute_pariticpation_ratio(x_new, u_new_mlp, nll_x, beta)

            mlp_ratios.append(ratio_mlp)
            mlp_part_ratios.append(participation_ratio_mlp)

        # calculate energy dft
        if n_steps % scheduled_dft == 0:
            u_new_dft = torch.zeros(n_chains)

            for i in range(n_chains):
                #molecule, logdetjac = coord_mapping.build_molecule_from_real_centered(
                #    x_new[i].reshape[1, -1], 
                #    int(isomer_new[i].item())
                #    )

                #u_ = calculator.calculate_potential_energy(
                #    molecule, 
                #    filename='ag6_'+str(dt)+'_'+str(i)+'.out'
                #)

                #u_new_dft[i] = coord_mapping.compute_energy_in_new_frame(u_, logdetjac*(-1))

                u_new_dft[i] = torch.tensor(-6.8+torch.rand(1)*0.5)

            # calculate ratio
            ratio_dft = compute_ratio(u_new_dft, u_init, nll_x, nll_x_init, beta)
            participation_ratio_dft = compute_pariticpation_ratio(x_new, u_new_dft, nll_x, beta)

            dft_ratios.append(ratio_dft)
            dft_part_ratios.append(participation_ratio_dft)

        u = torch.rand_like(ratio_mlp)

        acc_mlp = u < torch.min(ratio_mlp, torch.ones_like(ratio_mlp))

        x_new[~acc_mlp] = x_init[~acc_mlp]
        u_new_mlp[~acc_mlp] = u_init[~acc_mlp]

        x_init = x_new.clone().detach()
        u_init = u_new_mlp.clone().detach()

    all_ratios = {
        'ratios': (mlp_ratios, dft_ratios),
        'part_ratios': (mlp_part_ratios, dft_part_ratios)
    }

    return all_ratios
