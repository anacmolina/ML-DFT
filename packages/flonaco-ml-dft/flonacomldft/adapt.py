import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import torch
from torch.nn.utils import clip_grad_norm_
import flonacomldft.gaussian_utils
from flonacomldft.gaussian_utils import MoG
from flonacomldft.sampling import (
    run_MALA,
    run_metropolis,
    run_metromalangevin,
    compute_ESS
)


def run_mcmc_adapt(model, target, n_iter=10, lr=1e-1, bs=100,
          use_scheduler=False,
          step_schedule=10000,
          args_samp={'type': 'mhmalangevin'},
          estimate_tau=False,
          return_all_xs=True,
          save_splits=10,
          grad_clip=1e4):
    """"
    Main adpating/sampling function.

    Args:
        model (Realnvp_MLP)
        target (MoG, PhiFour)
        n_iter (int)
        lr (float): learning rate
        bs (int): batchsize
        use_scheduler (bool): if learning rate schedule should be used
        step_schedule (int): iteration frequency of schedule   
        args_samp (dict): ampling method 'mhmalangevin', 'malalangevin'
                    + kwargs for sampling method
        estimate_tau: estimates autocorrelation time
        return_all_xs: will return samples produced on the fly
        save_splits: number of snapshots saved during training
    """

    # setting up the loss
    def loss_func(x): return (model.nll(x) - target.U(x)).mean()

    # setting the sampling
    if 'langevin' in args_samp['type']:
        ## setting initialization for chain methods
        skip_burnin = False
        assert args_samp['n_tot'] <= bs
        
        if args_samp['x_init_samp'] is not None:
            x_init = args_samp['x_init_samp'][-args_samp['n_tot']:]
            skip_burnin = True
        elif isinstance(target, MoG) or isinstance(target, Croissants):
            x_init = torch.stack(target.means)
            x_init = x_init.repeat_interleave(
                int(args_samp['n_tot'] / len(target.means)), dim=0)
        else:
            raise NotImplementedError("TODO: Implement init within target class")

        x_init = x_init.detach().requires_grad_()

        ## setting samplimg functions
        if args_samp['type'] == 'mhmalangevin':

            def sample_func(bs, x_init=x_init, dt=100, beta=1, alpha=0, acc_rate=None):
                n_steps = int(bs / x_init.shape[0])
                x, acc = run_metromalangevin(
                    model, target, x_init, n_steps, dt * model.dim)
                kwargs['x_init'] = x[-1, ...].detach().requires_grad_()
                kwargs['acc_rate'] = (acc.cpu().numpy() * 1).mean()
                return x
    
        elif args_samp['type'] == 'malangevin':

            def sample_func(bs, x_init=x_init, dt=100, beta=1, acc_rate=None):
                n_steps = int(bs / x_init.shape[0])
                x, acc = run_MALA(
                    target, x_init, n_steps, dt=dt * model.dim)
                kwargs['x_init'] = x[-1, ...].detach().requires_grad_()
                kwargs['acc_rate'] = (acc.cpu().numpy() * 1).mean()
                return x

        kwargs = {'x_init': x_init,
                  'dt': args_samp['dt'],
                  'beta': args_samp['beta']}

        if not skip_burnin:
            bs_burnin = int(args_samp['n_steps_burnin'] * x_init.shape[0])
            start = time.time()
            n_steps = int(bs_burnin / x_init.shape[0])
            x, _ = run_MALA(target, x_init, n_steps, args_samp['dt'] * model.dim)
            kwargs['x_init'] = x[-1, ...].detach().requires_grad_()
            print('MALA burnin done! time: {:f}s'.format(time.time() - start))

    assert sample_func is not None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=step_schedule, 
                                                    gamma=0.5)

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(2, 5)
    axs = [fig.add_subplot(gs[a]) for a in range(10)]
    a = 0  # counter index for axs

    # logs
    xs = []
    losses = []
    models = [copy.deepcopy(model)]
    taus = []
    acc_rates = []
    acc_rates_mala = []
    grad_norms = []

    for t in range(n_iter):
        optimizer.zero_grad()

        x_ = sample_func(bs, **kwargs)

        x = x_.reshape(-1, model.dim).detach().requires_grad_()
        loss = loss_func(x)

        if return_all_xs or t % (n_iter / 10) == 0:
            xs.append(x_)

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

        if estimate_tau:
            tau = x.shape[0] * x.shape[1] / \
                np.mean(compute_ESS(x.detach().cpu()))
            taus.append(tau)

        x_last = x.clone()
        _, acc = run_metropolis(model, target, x_last, 1)
        acc_rate = (acc.cpu().numpy() * 1).mean()
        acc_rates.append(acc_rate.item())
        _, acc = run_MALA(target, x_last, 1,  dt=args_samp['dt'] * model.dim)
        acc_rate = (acc.cpu().numpy() * 1).mean()
        acc_rates_mala.append(acc_rate.item())

        #prints
        if t % (n_iter / save_splits) == 0 or n_iter <= save_splits:
            models.append(copy.deepcopy(model))
            print('t={:0.1e}'.format(t),
                  'Loss: {:3.2f}'.format(loss.item()), end='  \t')

            print('mh acc: {:0.2e}, mala acc: {:0.2e}'.format(acc_rates[-1],
                                                      acc_rates_mala[-1]),
                                                      end='\t')

            print('Gd: {:0.0e}'.format(total_norm), end='\t')

            for param_group in optimizer.param_groups:
                print('lr: {:0.2e}'.format(param_group['lr']), end='\t')

            print('')

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
            a += 1  # update counter of axes to plots

    to_return = {
        'model': model,
        'losses': losses,
        'xs': xs,
        'models': models,
        'taus': taus,
        'acc_rates': acc_rates,
        'acc_rates_mala': acc_rates_mala,
        'grad_norms': grad_norms,
    }

    return to_return
