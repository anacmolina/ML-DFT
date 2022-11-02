import copy
from pickletools import optimize

import torch
import numpy as np

import matplotlib.pyplot as plt

from torch.nn.utils import clip_grad_norm_

from flonacomldft.data_utils import (
    get_path
)

from flonacomldft.models import MLP, center_values

import sklearn.model_selection
import torch.optim as optim
import tqdm

def train_mlp(model, 
    input_val, 
    output_val, 
    n_iter=100, 
    lr=1e-4,
    bs=100,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
    retraining=False
):

    #mse
    def loss_func(x,y):
        return ((model(x) - y[:,None]) ** 2).mean()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_schedule, gamma=0.5
        )

    losses = []
    losses_val = []
    models = [copy.deepcopy(model)]
    grad_norms = []

    sk_seed = 42
    train_size = 0.8

    x = input_val.detach().requires_grad_().float()
    y = output_val.detach().requires_grad_().float()

    if retraining:
        x_centered = center_values(x, model.x_mean, model.x_centered_std)
        y_centered = center_values(y, model.y_mean, model.y_centered_std)
    else:
        x_centered, x_mean, x_centered_std = center_values(x)
        y_centered, y_mean, y_centered_std = center_values(y)

        means = [x_mean, y_mean]
        stds = [x_centered_std, y_centered_std]

        model.set_center_values(means, stds)

    arrays = [x_centered, y_centered]

    data_split = sklearn.model_selection.train_test_split(*arrays, test_size=None,
                                                      train_size=train_size,
                                                      random_state=sk_seed,
                                                      shuffle=True,
                                                      stratify=None)

    x_train, x_test, y_train, y_test = data_split

    pbar =  tqdm.tqdm(range(n_iter))

    for t in pbar:
        optimizer.zero_grad()

        loss = loss_func(x_train, y_train)

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
        losses_val.append(loss_func(x_test, y_test).item())
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

        if t % (n_iter / save_splits) == 0 or n_iter <= save_splits:
            models.append(copy.deepcopy(model))

    to_return = {
        'mlp_model': model,
        "dataset_split": data_split,
        "losses": losses,
        "losses_test": losses_val,
        "models": models,
        "grad_norms": grad_norms,
    }

    return to_return
