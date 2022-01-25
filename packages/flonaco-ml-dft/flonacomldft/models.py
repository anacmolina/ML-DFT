import torch
import torch.nn as nn # To build the NN (MLP)
import numpy as np


class MLP(nn.Module):
    def __init__(self, layerdims, activation=torch.relu, init_scale=None):
        super(MLP, self).__init__() # Access to all functions
        self.layerdims = layerdims   # Setting parameters of the NN 
        self.activation = activation  # Layers, activation. It's defined in the initialization
        linears = [nn.Linear(layerdims[i], layerdims[i + 1]) for i in range(len(layerdims) - 1)] 
        # Applies a linear transformation
        # Build the NN architecture

        if init_scale is not None:
            for l, layer in enumerate(linears):
                # Weigths initializations with normal distribution
                # Bias initialializations zeros
                torch.nn.init.normal_(layer.weight, 
                                      std=init_scale/np.sqrt(layerdims[l]))
                torch.nn.init.zeros_(layer.bias)

        # 
        self.linears = nn.ModuleList(linears)

    # Forward propagation of the NN
    # Uses the activation function to do the calculation
    def forward(self, x):
        layers = list(enumerate(self.linears))
        for _, l in layers[:-1]:
            x = self.activation(l(x))
        y = layers[-1][1](x)
        return y
