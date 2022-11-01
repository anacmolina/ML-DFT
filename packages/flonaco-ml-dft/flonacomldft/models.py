import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):

    def __init__(self, layerdims, activation=torch.relu, init_scale=None, center_data=False):
        super(MLP, self).__init__()

        self.LOCK = False
        self.center_data = center_data
        
        self.layerdims = layerdims
        self.activation = activation
        linears = [nn.Linear(layerdims[i], layerdims[i + 1]) for i in range(len(layerdims) - 1)]
        
        if init_scale is not None:
            for l, layer in enumerate(linears):
                torch.nn.init.normal_(layer.weight, 
                                      std=init_scale/np.sqrt(layerdims[l]))
                torch.nn.init.zeros_(layer.bias)

        self.linears = nn.ModuleList(linears)

    def forward(self, x):

        layers = list(enumerate(self.linears))
        for _, l in layers[:-1]:
            x = self.activation(l(x))
        y = layers[-1][1](x)

        return y

    def set_center_data(self, value):
        self.center_data(value)

    def set_center_values(self, means, stds):

        if self.center_data:

            if (not self.LOCK) and (means is not None) and (stds is not None):
                
                self.x_mean = means[0]
                self.y_mean = means[1]
                self.x_centered_std = stds[0]
                self.y_centered_std = stds[1]
                
                self.LOCK = True

    def predict(self, x):

        if self.LOCK:
            x = x - self.x_mean
            x = x/self.x_centered_std
            
            y = self.forward(x)

        
            y = y * self.y_centered_std
            y = y + self.y_mean

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



""" 
class MLP(nn.Module):
     def __init__(self, layerdims, activation=torch.relu, init_scale=None):
        super(MLP, self).__init__()
        self.layerdims = layerdims
        self.activation = activation
        linears = [nn.Linear(layerdims[i], layerdims[i + 1]) for i in range(len(layerdims) - 1)]
        
        if init_scale is not None:
            for l, layer in enumerate(linears):
                torch.nn.init.normal_(layer.weight, 
                                      std=init_scale/np.sqrt(layerdims[l]))
                torch.nn.init.zeros_(layer.bias)

        self.linears = nn.ModuleList(linears)

     def forward(self, x):
        layers = list(enumerate(self.linears))
        for _, l in layers[:-1]:
            x = self.activation(l(x))
        y = layers[-1][1](x)
        return y 
"""


""" class Uncentered_MLP:
    def __init__(self, model_info):
        self.model = model_info['mlp_model']
        self.x_mean = model_info['x_mean']
        self.x_centered_std = model_info['x_centered_std']
        self.y_mean = model_info['y_mean']
        self.y_centered_std = model_info['y_centered_std']
    
    def __call__(self, x):

        x = x - self.x_mean
        x = x/self.x_centered_std

        y = self.model(x)
        y = y * self.y_centered_std
        y = y + self.y_mean

        return y """