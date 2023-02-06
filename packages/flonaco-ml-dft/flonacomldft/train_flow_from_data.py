import copy
import tqdm

import torch
from torch.nn.utils import clip_grad_norm_
from ray import tune

from ase.parallel import parprint as print
from flonacomldft.sampling import run_metropolis

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
        isomer=train[0, 14].to(torch.int64).item()
        T=T
        n_chains=n_chains
        ratios = []
        acc_rates = []
        from flonacomldft.utils.io_utils import load_pickle_file 
        mlp = load_pickle_file("models/is{:d}_mlp_dic_training.pkl".format(isomer))['model']

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

        if compute_ratio_acc:

            n_chains = n_chains
            n_steps = 1

            """
            _ = run_metropolis(model=model,
                        init=test[:n_chains],
                        n_chains=n_chains,
                        n_steps=n_steps,
                        n_run="",
                        energy_type='mlp', #'dft',
                        frac_dft=None,
                        mlp_models=mlp,
                        mixture=False,
                        T=300,
                        with_tqdm=False,
                    )
            acc_rate = (_['accs'].cpu().numpy() * 1).mean()
            acc_rates.append(acc_rate)
            """
            # TODO: get both, acceptance and ratio
            def get_ratio_acc(
                        model,
                        init,
                        n_chains,
                        n_steps,
                        mlp_model,
                        T=T,
                ):  

                assert init.shape[0] == n_chains

                kb = 8.617333262e-5
                beta = 1 / (kb * T)

                x_init = init[:, :12]
                u_init = init[:, 13]
                isomer_init = init[:, 14]

                for dt in range(n_steps):

                    x_new = model.sample(n_chains)
                    isomer_new = isomer_init

                    x_new = x_new.clone().detach().float()
                    isomer_new = isomer_new.clone().detach().float()

                    nll_x = model.nll(x_new)
                    nll_x_init = model.nll(x_init)

                    u_new = torch.zeros((n_chains, 1))
                    u_new = mlp_model(x_new)
                    u_new = u_new.squeeze().float()

                    #print('energies: ', u_new, u_init)
                    #print('nll: ', nll_x, nll_x_init)

                    ratio = -beta * u_new + nll_x
                    ratio += beta * u_init - nll_x_init
                    ratio = torch.exp(ratio)

                    u = torch.rand_like(ratio)
                    ratio = torch.min(ratio, torch.ones_like(ratio))
                    acc = u < ratio

                return ratio, acc

            ratio, acc_rate = get_ratio_acc(model=model,
                    init=test[:n_chains],
                    n_chains=n_chains,
                    n_steps=n_steps,
                    mlp_model=mlp,
                    T=300,)

            #print('ratio: ', acc_rate, (acc_rate.cpu().detach().numpy() * 1).mean())
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
                "t={:0.1e}".format(t), "Loss: {:3.2f}".format(loss.item()), end="  \t"
            )

            if compute_ratio_acc:
                print("ratio: {:.3f}".format(ratio), end="\t")

                print("acc: {:.3f}".format(acc_rate), end="\t")

            print("Gd: {:0.0e}".format(total_norm), end="\t")

            for param_group in optimizer.param_groups:
                print("lr: {:0.2e}".format(param_group["lr"]), end="\t")

            print("")
            if use_tune:
                tune.report({"loss":loss.item(), "grad_norm":total_norm})

    to_return = {
        "model": model,
        "losses": (losses_train, losses_test),
        "models": models,
        "grad_norms": grad_norms,
    }

    if compute_ratio_acc:
        to_return["acc_rates"]= acc_rates,
        to_return["ratios"]: ratios

    return to_return