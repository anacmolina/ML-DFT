import copy
import torch
from torch.nn.utils import clip_grad_norm_

from ase.parallel import parprint as print


def train_flow(
    model,
    x_train,
    n_iter=1000,
    lr=5e-3,
    # bs=100,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
):
    """ 
    Args:
        model (Realnvp_MLP)
        x_train (tensor of float)
        n_iter (int)
        lr (float): learning rate
        # bs (int): batchsize
        use_scheduler (bool): if learning rate schedule should be used
        step_schedule (int): iteration frequency of schedule
        save_splits: number of snapshots saved during training
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

    # logs
    losses = []
    models = [copy.deepcopy(model)]
    grad_norms = []

    x = x_train.detach().requires_grad_()

    for t in range(n_iter):
        optimizer.zero_grad()

        loss = loss_func(x)

        if torch.isinf(loss).any():
            print("Stopped because loss became inf!")
            return model, losses

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # logs
        losses.append(loss.item())

        if t % (n_iter / 100) == 0:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)

        if use_scheduler:
            scheduler.step()

        # prints
        if t % (n_iter / save_splits) == 0 or n_iter <= save_splits:
            models.append(copy.deepcopy(model))
            print(
                "t={:0.1e}".format(t), "Loss: {:3.2f}".format(loss.item()), end="  \t"
            )

            print("Gd: {:0.0e}".format(total_norm), end="\t")

            for param_group in optimizer.param_groups:
                print("lr: {:0.2e}".format(param_group["lr"]), end="\t")

            print("")

    to_return = {
        "model": model,
        "losses": losses,
        "models": models,
        "grad_norms": grad_norms,
    }

    return to_return
