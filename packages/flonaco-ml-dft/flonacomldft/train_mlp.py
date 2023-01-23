import copy
import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from flonacomldft.internal_coordinates import Coordinates_mapping

def train_mlp(
    model,
    train,
    test,
    isomer,
    n_iter=100,
    lr=1e-4,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    with_tqdm=False,
):

    # mse: loss function with data centered
    def loss_func(x, y):
        # x and y centered
        return ((model(x).squeeze() - y) ** 2).mean()

    # setting the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_schedule, gamma=0.5
        )

    zs_train, logdetjac_train, us_train = train[:, :12], train[:, 12], train[:, 13]
    zs_test, logdetjac_test, us_test = train[:, :12], train[:, 12], train[:, 13]  

    coord_mapping = Coordinates_mapping()

    zs_train, logdetjac_train, us_train = coord_mapping.get_real_centered_from_internal(zs_train, logdetjac_train, isomer=isomer, energies=us_train)
    zs_test, logdetjac_test, us_test = coord_mapping.get_real_centered_from_internal(zs_test, logdetjac_test, isomer=isomer, energies=us_test)

    if with_tqdm:
        pbar = tqdm.tqdm(range(n_iter))
    else:
        pbar = range(n_iter)


    # logs
    losses_train = []
    losses_test = []
    grad_norms = []
    models = [copy.deepcopy(model)]

    for t in pbar:
        optimizer.zero_grad()

        loss = loss_func(zs_train, us_train)

        if torch.isinf(loss).any():
            print("Stopped because loss became inf!")
            return model, losses_train

        loss.backward(retain_graph=True)
        optimizer.step()

        losses_train.append(loss.item())
        losses_test.append(loss_func(zs_test, us_test).item())


        if with_tqdm:
            pbar.set_description(f"Loss train: {losses_train[-1]:.4f}, loss test: {losses_test[-1]:.4f}")

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

    to_return = {
        "model": model,
        "losses": (losses_train, losses_test),
        "models": models,
        "grad_norms": grad_norms,
    }

    return to_return