import copy
import tqdm

import torch
from torch.nn.utils import clip_grad_norm_

from ase.parallel import parprint as print
# from ray import tune

from flonacomldft.sampling import run_metropolis


def train_flow(
    model,
    x_train,
    x_test,
    u_test,
    isomer,
    n_iter=1000,
    lr=5e-3,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
    with_tqdm=False,

    use_tune=False,
    metrics=None, 
):
    """
    Args:
        model (Realnvp_MLP)
        x_train (tensor of float)
        x_test (tensor of float)
        n_iter (int)
        lr (float): learning rate
        use_scheduler (bool): if learning rate schedule should be used
        step_schedule (int): iteration frequency of schedule
        save_splits: number of snapshots saved during training
        grad_clip (float): gradient clipping
        use_tune (bool): whether a tuner from Ray package has been initialized
    """

    # setting up the loss
    def loss_func(x):
        return (model.nll(x)).mean()

    # setting the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_schedule, gamma=0.5
        )

    from flonacomldft.utils.io_utils import load_pickle_file 
    mlp = load_pickle_file("models/is{:d}_mlp_dic_training.pkl".format(isomer+1))['model']

    # logs
    losses_train = []
    losses_test = []
    acc_rates = []
    models = [copy.deepcopy(model)]
    grad_norms = []

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

        #TODO: COMPUTE WITH DFT OR MLP (ALTERNATE), DON'T COMPUTE FOR ALL CASES?
        n_chains = 50
        n_steps = 50
        x_samp = x_train.clone()[:n_chains] + model.centering_args['mean_out']
        u_samp = u_test.clone()[:n_chains]

        _ = run_metropolis(model=model,
                    x_init=x_samp,
                    u_init=u_samp,
                    isomer_init=torch.full((n_chains, 1), isomer),
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

        if with_tqdm:
            pbar.set_description(f"Loss: {losses_train[-1]:.4f}")

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

            print("accs: {:.3f}".format(acc_rate), end="\t")

            print("Gd: {:0.0e}".format(total_norm), end="\t")

            for param_group in optimizer.param_groups:
                print("lr: {:0.2e}".format(param_group["lr"]), end="\t")

            print("")
            # if use_tune:
            #     tune.report({"loss":loss.item(), "grad_norm":total_norm})

    to_return = {
        "model": model,
        "losses": (losses_train, losses_test),
        "acc_rates": acc_rates,
        "models": models,
        "grad_norms": grad_norms,
    }

    return to_return
