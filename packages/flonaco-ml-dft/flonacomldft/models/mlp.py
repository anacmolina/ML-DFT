import torch
import torch.nn as nn
import numpy as np

from flonacomldft.internal_coordinates import Coordinates_mapping

class MLP(nn.Module):
    def __init__(self, layerdims, activation=torch.relu, init_scale=None):
        super(MLP, self).__init__()
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



