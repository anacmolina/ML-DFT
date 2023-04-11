import copy
import tqdm

import torch
from torch.nn.utils import clip_grad_norm_
# from ray import tune

from ase.parallel import parprint as print

kb = 8.617333262e-5

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

    beta = 1 / (kb * T)

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

# def get_ratio_acc(
#             model,
#             init,
#             n_chains,
#             n_steps,
#             training_iteration,
#             mlp_model=None,
#             T=300,
#     ):
# 
#     assert init.shape[0] == n_chains
# 
#     kb = 8.617333262e-5
#     beta = 1 / (kb * T)
# 
#     x_init = init[:, :12]
#     isomer_init = init[:, 13]
# 
#     x_new = model.sample(n_chains)
#     isomer_new = isomer_init
# 
#     x_new = x_new.clone().detach().float()
#     isomer_new = isomer_new.clone().detach().float()
# 
#     nll_x = model.nll(x_new)
#     nll_x_init = model.nll(x_init)
# 
#     u_new = torch.zeros((n_chains, 1))
# 
#     if mlp_model is None:
#         u_init = init[:, 12]
# 
#         from flonacomldft.dft_calculator import DFTCalculator
#         from flonacomldft.internal_coordinates import Coordinates_mapping
#         
#         coord_maps = Coordinates_mapping()
#         calculator = DFTCalculator()
#         calculator.initialize_calculator()
# 
#         for i in range(len(x_new)):
#             molecule, logdetjac = coord_maps.build_molecule_from_real_centered(x_new[i].reshape(1, -1), isomer_new[i].item())
#             u_ = calculator.calculate_potential_energy(
#                                 molecule, 
#                                 filename='ag6_'+str(training_iteration)+'_'+str(i)+'.out'
#                                                 )
#             u_new[i] = coord_maps.compute_energy_in_new_frame(u_, logdetjac*(-1))
# 
#     else:
#         u_init = mlp_model(x_init)
#         u_init = u_init.squeeze().float()
#         u_new = mlp_model(x_new)
#         u_new = u_new.squeeze().float()
# 
#     #u_init = u_init.detach()
#     #u_new = u_new.detach()
# 
#     ratio = -beta * u_new + nll_x
#     ratio += beta * u_init - nll_x_init
#     ratio = torch.exp(ratio)
# 
#     u = torch.rand_like(ratio)
#     ratio = torch.min(ratio, torch.ones_like(ratio))
#     acc = u < ratio
# 
#     return ratio, acc

def train_flow(
    model,
    train,
    test,
    n_iter=100,
    lr=5e-3,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
    with_tqdm=False,
    use_tune=False,
    compute_ratio_acc=True,
    mlp_model=None,
    n_chains=100,
    T=300,
):

    # setting up the loss
    def loss_func(x):
        return (model.nll(x)).mean()

    # setting the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_schedule, gamma=0.5
        )
        
    x_train = train[:, :12]
    x_test = test[:, :12]

    # logs
    losses_train = []
    losses_test = []
    models = [copy.deepcopy(model)]
    grad_norms = []

    if compute_ratio_acc:
        isomer=train[0, 13].to(torch.int64).item()
        T=T
        n_chains=n_chains
        ratios = []
        acc_rates = []
        #from flonacomldft.utils.io_utils import load_pickle_file 
        #mlp = load_pickle_file("models/is{:d}_mlp_dic_training.pkl".format(isomer))['model']

    x = x_train.detach().requires_grad_()

    if with_tqdm:
        pbar = tqdm.tqdm(range(n_iter))
    else:
        pbar = range(n_iter)

    for t in pbar:
        optimizer.zero_grad()

        loss = loss_func(x)

        if torch.isinf(loss).any():
            print("Stopped because loss became inf!")
            return model, losses_train

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        losses_train.append(loss.item())
        losses_test.append(loss_func(x_test).item())

        if with_tqdm:
            pbar.set_description(f"Loss: {losses_train[-1]:.4f}")

        if compute_ratio_acc==True and t % (n_iter / save_splits) == 0:

            n_chains = n_chains
            ratio, acc_rate = get_ratio_acc(model=model,
                    init=test[:n_chains],
                    n_chains=n_chains,
                    training_step=t,
                    mlp_model=mlp_model,
                    T=T)

            ratio = (ratio.cpu().detach().numpy() * 1).mean()
            ratios.append(ratio)
        
            acc_rate = (acc_rate.cpu().detach().numpy() * 1).mean()
            acc_rates.append(acc_rate)

        if t % (n_iter / 100) == 0:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)

        if use_scheduler:
            scheduler.step()

        if t % (n_iter / save_splits) == 0 or n_iter <= save_splits:
            models.append(copy.deepcopy(model))

            # prints

            print(
                "t={:0.1e}".format(t), "loss: {:3.2f}".format(loss.item()), end="  \t"
            )

            if compute_ratio_acc:
                print("ratio: {:.3f}".format(ratio), end="\t")

                print("acc: {:.3f}".format(acc_rate), end="\t")

            print("Gd: {:0.0e}".format(total_norm), end="\t")

            for param_group in optimizer.param_groups:
                print("lr: {:0.2e}".format(param_group["lr"]), end="\t")

            print("")
            # if use_tune:
            #     tune.report({"loss":loss.item(), "grad_norm":total_norm})

    to_return = {
        "model": model,
        "losses": (losses_train, losses_test),
        "models": models,
        "grad_norms": grad_norms,
    }

    if compute_ratio_acc:
        to_return["acc_rates"] = acc_rates
        to_return["ratios"] = ratios

    return to_return