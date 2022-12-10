import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

import tqdm
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import sklearn.model_selection

from flonacomldft.models import MLP, center_values

def train_mlp(model, 
    x_train,
    y_train,
    x_test,
    y_test, 
    n_iter=100, 
    lr=1e-4,
    use_scheduler=False,
    step_schedule=100,
    grad_clip=1e4,
):

    #mse
    def loss_func(x,y):
        # x and y centered
        return ((model(x) - y[:,None]) ** 2).mean()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_schedule, gamma=0.5
        )

    losses = []
    losses_val = []
    grad_norms = []
    model_init = copy.deepcopy(model)

    if model.has_centered:
        x_train_centered = center_values(x_train, model.x_mean, model.x_centered_std)
        y_train_centered = center_values(y_train, model.y_mean, model.y_centered_std)
        x_test_centered = center_values(x_test, model.x_mean, model.x_centered_std)
        y_test_centered = center_values(y_test, model.y_mean, model.y_centered_std)
    else:
        x_train_centered, x_mean, x_centered_std = center_values(x_train)
        y_train_centered, y_mean, y_centered_std = center_values(y_train)
        x_test_centered = center_values(x_test, x_mean, x_centered_std)
        y_test_centered = center_values(y_test, y_mean, y_centered_std)

        means = [x_mean, y_mean]
        stds = [x_centered_std, y_centered_std]

        model.set_center_values(means, stds)

    pbar =  tqdm.tqdm(range(n_iter))

    for t in pbar:
        optimizer.zero_grad()

        loss = loss_func(x_train_centered, y_train_centered)

        # In case we are running out of memory
        # if return_all_xs or t % (n_iter / 10) == 0:
        #    xs.append(x_)

        if torch.isinf(loss).any():
            print("Stopped because loss became inf!")
            return model, losses
        
        loss.backward(retain_graph=True)
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        losses.append(loss.item())
        losses_val.append(loss_func(x_test_centered, y_test_centered).item())
        pbar.set_description(f'Loss: {losses[-1]:.4f}')

        if t % (n_iter / 100) == 0:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)

        if use_scheduler:
            scheduler.step()

        #if t % (n_iter / save_splits) == 0 or n_iter <= save_splits:
        #    models.append(copy.deepcopy(model))

    to_return = {
        'model': model,
        "dataset": (x_train_centered, y_train_centered, x_test_centered, y_test_centered),
        "losses": losses,
        "losses_test": losses_val,
        "model_init": model_init,
        "grad_norms": grad_norms,
    }

    return to_return
