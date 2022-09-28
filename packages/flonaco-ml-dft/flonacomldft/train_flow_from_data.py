import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import torch
from torch.nn.utils import clip_grad_norm_

from ase.parallel import parprint as print

import flonacomldft.gaussian_utils
from flonacomldft.gaussian_utils import MoG
from flonacomldft.sampling import (
    #run_MALA,
    run_metropolis,
    #run_metromalangevin,
    compute_ESS
)


def train(model, x_train, n_iter=10, lr=1e-1, bs=100,
          use_scheduler=False,
          step_schedule=10000,
          args_loss={'type': 'fwd', 'samp': 'direct'},
          estimate_tau=False,
          return_all_xs=True,
          save_splits=10,
          grad_clip=1e4):
    """"
    Args:
        model (Realnvp_MLP)
        n_iter (int)
        lr (float): learning rate
        bs (int): batchsize
        use_scheduler (bool): if learning rate schedule should be used
        step_schedule (int): iteration frequency of schedule   
        args_loss (dict): 
                    'type' - loss type 'fwd', 'bwd', 'js'
                    'samp' - sampling method 'langevin', 'direct', 'mhlangevin' 
                    + kwargs for sampling method
                    Note that not all combinations are possible
                    depending on target etc.
        args_stop (dict): {'acc': x} with x in [0,1] Metropolis acceptance
                    threshold to stop train
        estimate_tau: estimates autocorrelation time
        return_all_xs: will return samples produced on the fly
        save_splits: number of snapshots saved during training
    """

    # setting up the loss
    def loss_func(x): return (model.nll(x)).mean()

    # setting the optimizer    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=step_schedule, 
                                                    gamma=0.5)

    #fig = plt.figure(figsize=(15, 5))
    #gs = gridspec.GridSpec(2, 5)
    #axs = [fig.add_subplot(gs[a]) for a in range(10)]
    #a = 0  # counter index for axs

    # logs
    xs = []
    losses = []
    models = [copy.deepcopy(model)]
    #taus = []
    #acc_rates = [] 
    #acc_rates_mala = []
    grad_norms = []

    x = x_train.detach().requires_grad_()

    for t in range(n_iter):
        optimizer.zero_grad()

        loss = loss_func(x)

        ### In case we are running out of memory
        #if return_all_xs or t % (n_iter / 10) == 0:
        #    xs.append(x_)

        if torch.isinf(loss).any():
            print('Stopped because loss became inf!')
            return model, losses, xs

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        #logs
        losses.append(loss.item())

        if t % (n_iter / 100) == 0:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5 
            grad_norms.append(total_norm)

        if use_scheduler:
            scheduler.step()

        #if estimate_tau:
        #    tau = x.shape[0] * x.shape[1] / \
        #        np.mean(compute_ESS(x.detach().cpu()))
        #    taus.append(tau)

        #x_last = x.clone()
        #_, acc = run_metropolis(model, target, x_last, 1)
        #acc_rate = (acc.cpu().numpy() * 1).mean()
        #acc_rates.append(acc_rate.item())
        #_, acc = run_MALA(target, x_last, 1,  dt=args_loss['dt'] * model.dim)
        #acc_rate = (acc.cpu().numpy() * 1).mean()
        #acc_rates_mala.append(acc_rate.item())

        #prints
        if t % (n_iter / save_splits) == 0 or n_iter <= save_splits:
            models.append(copy.deepcopy(model))
            print('t={:0.1e}'.format(t),
                  'Loss: {:3.2f}'.format(loss.item()), end='  \t')

            #print('mh acc: {:0.2e}, mala acc: {:0.2e}'.format(acc_rates[-1],
            #                                          acc_rates_mala[-1]),
            #                                          end='\t')

            print('Gd: {:0.0e}'.format(total_norm), end='\t')

            for param_group in optimizer.param_groups:
                print('lr: {:0.2e}'.format(param_group['lr']), end='\t')

            print('')
        """
        if t % (n_iter / 10) == 0:
            if model.dim == 2:
                assert isinstance(target, MoG) or isinstance(target, Croissants)
                x_min = -10 if isinstance(target, MoG) else -0.5
                x_max = 10 if isinstance(target, MoG) else 0.5
                flonacomldft.gaussian_utils.plot_2d_level(model, ax=axs[a],
                                                    title='t= ' + str(t),
                                                    x_min=x_min, x_max=x_max)

                plt.scatter(x[:, 0].detach().cpu(),
                            x[:, 1].detach().cpu(), s=1., alpha=0.1)

            plt.tight_layout()
            a += 1  # update counter of axes to plots"""

    to_return = {
        'model': model,
        'losses': losses,
        'xs': xs,
        'models': models,
        #'taus': taus,
        #'acc_rates': acc_rates,
        #'acc_rates_mala': acc_rates_mala,
        'grad_norms': grad_norms,
    }

    return to_return
