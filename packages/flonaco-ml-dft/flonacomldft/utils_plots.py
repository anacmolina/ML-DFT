import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from flonacomldft.real_nvp_mlp import RealNVP_MLP


n_points=1000
x_min=-5
x_max=5
y_min=None
y_max=None

def plot_2d_level(log_prob, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                  n_points=n_points, device='cpu', ax=None, title=''):

    x_range = torch.linspace(x_min, x_max, n_points, device=device)

    if y_min is None:
        y_range = x_range.clone()
    else:
        y_range = torch.linspace(y_min, y_max, n_points, device=device)

    grid = torch.meshgrid(x_range, y_range)
    xys = torch.stack(grid).reshape(2, n_points ** 2).T.to(device)

    Us = - log_prob(xys).reshape(n_points, n_points).T.detach().cpu().numpy()

    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)
    plt.contourf(x_range, y_range, np.exp(- Us[::-1]), 10, cmap='GnBu')
    plt.title(title)

    plt.tight_layout()
    plt.show(block=False)
    return Us