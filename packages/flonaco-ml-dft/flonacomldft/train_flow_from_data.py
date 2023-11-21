# import libraries

import time
import copy
import tqdm
import torch
from torch.nn.utils import clip_grad_norm_

from ase.parallel import parprint as print

#TODO: add docstring
#TODO: add device

def train_flow(
        model,
        train,
        n_iter,
        lr,
        bs,
        test=None,
        use_scheduler=False,
        step_scheduler=10,
        save_splits=1,
        grad_clip=1e4,
        with_tqdm=False,
        n_partial_loss=10,
        dim=12
    ):

    def loss_func(x):
        return (model.nll(x)).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=step_scheduler, 
                                                    gamma=0.5)

    if save_splits > 1:
        models = [copy.deepcopy(model)]

    train_losses = []
    grad_norms = []
    time_step = []

    if test is not None:
        test_losses = []

    if with_tqdm:
        
        pbar = tqdm.tqdm(range(n_iter))
    
    else:
        
        pbar = range(n_iter)
        print('Epoch \t\t Train Lr \t Loss \t\t Grad norm')


    x = train[:, :dim].clone().detach()
    permutation = torch.randperm(x.shape[0])

    for t in pbar:

        for k in range(0, x.shape[0], bs):

            indexes = permutation[k:k+bs]
            x_batch = x[indexes].clone()

            optimizer.zero_grad()
            loss = loss_func(x_batch)

            if torch.isinf(loss).any():
                print('Stopped because loss became inf!')
                to_return = {'model': model, 
                        'train_losses': train_losses}
                if test is not None:
                    to_return['test_losses'] = test_losses
                
                return to_return
            
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

        train_losses.append(loss.item())
        time_step.append(time.time())

        if test is not None:
            test_losses.append(loss_func(test[:, :dim]).item())

        if t % (n_iter / 100) == 0:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)

        if use_scheduler:
            scheduler.step()

        if with_tqdm == False:

            if t % (n_iter // n_partial_loss) == 0:

                for param_group in optimizer.param_groups:
                    lr_ = param_group['lr']

                print('{:0.1e} \t {:0.2e} \t {:3.2e} \t {:0.0e}'.format(t, lr_, train_losses[-1], grad_norms[-1]))

        else:

            pbar.set_description('Loss: {:.4f}'.format(train_losses[-1]))

        if save_splits > 1:
            if t % (n_iter // save_splits) == 0:
                models.append(copy.deepcopy(model))

    to_return = {
        'model': model,
        'train_losses': train_losses,
        'grad_norms': grad_norms,
        'time_step': time_step,
    }     

    if save_splits > 1:
        to_return['models'] = models

    if test is not None:
        to_return['test_losses'] = test_losses

    return to_return