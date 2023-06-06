from torch._C import device
import numpy as np
import torch
import torch.nn as nn

class Mixture(nn.Module):
    def __init__(self, models, init_weights=None, device='cpu'):
        super(Mixture, self).__init__()

        self.device = device
        self.dim = models[0].dim

        self.models = models

        if init_weights is None:
            self.weights = torch.ones((len(models),), device=device) / len(models)
        else:
            #self.weights = torch.tensor(init_weights).to(device)
            self.weights = init_weights.clone().detach().requires_grad_(True).to(device=device)

        self.weights.requires_grad_()

    def sample(self, n, shuffle=True, return_mus=False):

        a = torch.multinomial(self.weights, n, replacement=True)
        mus, counts = a.unique(return_counts=True)   

        xs = []
        cs = []
        for mu, count in zip(mus, counts):
            xs.append(self.models[mu].sample(count))
            cs += [mu.item()] * count

        x = torch.cat(xs)

        if shuffle:
            perm = torch.randperm(n)
            x = x[perm, :].float()
            cs = torch.tensor(cs)[perm].float()

        if return_mus:
            return x, cs
        else:
            return x

    def nll(self, x):
        args = []
        for model, weight in zip(self.models, self.weights):
            args.append(- model.nll(x) + torch.log(weight))

        return - torch.logsumexp(torch.stack(args), 0)

    def U(self, x):
        return self.nll(x)

def get_models(training):
    """
    Get the models for the mixture from a list of dictionnaries storing the models.
    """

    if training is None:
        return [None, None]
    else:
        return [training[0]['model'], training[1]['model']]
