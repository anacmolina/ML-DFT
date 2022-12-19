import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, layerdims, activation=torch.relu, init_scale=None):
        super(MLP, self).__init__()
        self.has_centered = False
        self.layerdims = layerdims
        self.activation = activation
        linears = [
            nn.Linear(layerdims[i], layerdims[i + 1]) for i in range(len(layerdims) - 1)
        ]

        if init_scale is not None:
            for l, layer in enumerate(linears):
                torch.nn.init.normal_(
                    layer.weight, std=init_scale / np.sqrt(layerdims[l])
                )
                torch.nn.init.zeros_(layer.bias)

        self.linears = nn.ModuleList(linears)

    def forward(self, x):
        """
        Predicts the centered y value for a given the centered x value
        """
        layers = list(enumerate(self.linears))
        for _, l in layers[:-1]:
            x = self.activation(l(x))
        y = layers[-1][1](x)
        return y

    def set_center_values(self, means, stds):

        self.x_mean = means[0]
        self.y_mean = means[1]
        self.x_centered_std = stds[0]
        self.y_centered_std = stds[1]

        self.has_centered = True

    def predict(self, x):
        """"
        Predicts the uncentered y value for a given the uncentered x value
        """

        x = x - self.x_mean
        x = x / self.x_centered_std

        y = self.forward(x)

        y = y * self.y_centered_std
        y = y + self.y_mean

        self.center = True

        return y


def center_values(x, x_mean=None, x_centered_std=None):

    if (x_mean is None) and (x_centered_std is None):
        x_mean = x.mean(0)
        x_centered = x - x.mean(0)

        x_centered_std = x_centered.std(0)
        x_centered = x_centered / x_centered.std(0)

        return x_centered, x_mean, x_centered_std

    else:
        x_centered = x - x_mean
        x_centered = x_centered / x_centered_std

        return x_centered

