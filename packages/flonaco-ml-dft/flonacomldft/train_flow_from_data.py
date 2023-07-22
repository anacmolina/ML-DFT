# import libraries

import time
import copy
import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
# from ray import tune

from ase.parallel import parprint as print

# flow training
def train_flow(
    model,
    train,
    test,
    dim=12,
    n_iter=100,
    lr=5e-4,
    batch_size=512,
    use_scheduler=False,
    step_schedule=10,
    save_splits=10,
    grad_clip=1e4,
    with_tqdm=False,
    use_tune=False,
    compute_part_ratio=False,
    energy_type='emt',
    mlp_model=None,
    n_prop=100,
    path=None,
):
    """Train a flow model on a dataset.

    Args:
        model: flow model
        train: training dataset
        test: test dataset (TODO: remove)
        n_iter: number of iterations
        lr: learning rate
        use_scheduler: use scheduler
        step_schedule: step size for scheduler
        save_splits: number of splits to save
        grad_clip: gradient clipping
        with_tqdm: use tqdm
        use_tune: use tune

    Returns:
        model: trained flow model
        losses: losses
        models: models
        grad_norms: gradient norms
    """
    

    # loss function 
    def loss_func(x):
        return (model.nll(x)).mean() # negative log likelihood

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_schedule, gamma=0.5
        )

    # input data   
    x_train = train[:, :dim]
    x_test = test[:, :dim]

    print('COMPUTE PARTICIPATION RATIO: ', compute_part_ratio)

    if compute_part_ratio:
        from flonacomldft.utils.diagnostics import Target_Log_Prob
        from flonacomldft.utils.diagnostics import get_participation_ratio

        mode_label = train[:, dim+1].unique().int()
        
        energy_type = energy_type
        mlp_model = mlp_model

        part_ratios = []
        time_part_ratios = []

    # logs
    losses_train = []
    losses_test = []
    models = [copy.deepcopy(model)]
    grad_norms = []

    x = x_train.clone().detach().requires_grad_(True)

    # print loss bar
    if with_tqdm:
        pbar = tqdm.tqdm(range(n_iter))
    else:
        pbar = range(n_iter) 

    time_epoch = []
    time_batch = []
    time_start = time.time()
    print("train shape: ", x.shape)

    permutation = torch.randperm(x.shape[0])

    # training loop
    for t in pbar:

        for  k in range(0, x.shape[0], batch_size):
            indexes = permutation[k:k+batch_size]
            batch_x = x[indexes]

            optimizer.zero_grad()
            loss = loss_func(batch_x)

            if torch.isinf(loss).any():
                print("Stopped because loss became inf!")
                return model, losses_train

            # backpropagation
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            time_batch.append(time.time())

        # appending logs
        losses_train.append(loss.item())
        losses_test.append(loss_func(x_test).item())
        time_epoch.append(time.time())

        if with_tqdm:
            pbar.set_description(f"Loss: {losses_train[-1]:.4f}")
        
        # gradient norms
        if t % (n_iter / 100) == 0:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)

        # learning rate scheduler
        if use_scheduler:
            scheduler.step()

        # saving models and printing logs
        if t % (n_iter / save_splits) == 0 or n_iter <= save_splits:
            models.append(copy.deepcopy(model))

            # prints

            print(
                "time:{:.2f}".format(time.time()-time_start), "\t epoch={:0.1e}".format(t), "\t loss: {:3.2f}".format(loss.item()), end="\t"
            )

            print("Gd: {:0.0e}".format(total_norm), end="\t")

            for param_group in optimizer.param_groups:
                if compute_part_ratio: 
                    line="\t"
                else:
                    line="\n"
                print("lr: {:0.2e}".format(param_group["lr"]), end=line)

            # if use_tune:
            #     tune.report({"loss":loss.item(), "grad_norm":total_norm})

            if compute_part_ratio:

                if energy_type == 'dft':
                    import gpaw.mpi as mpi
                    path = path+'/DFTComputations_{:d}'.format(t)
                else:
                    path = None
                    
                target_log_prob = Target_Log_Prob(energy_type=energy_type, mode_label=mode_label, mlp_model=mlp_model, folder=path).target_log_prob
                part_ratio = get_participation_ratio(model, target_log_prob, n_prop=n_prop)

                part_ratios.append(part_ratio)

                if energy_type == 'dft':
                    rank = mpi.rank
                    if rank == 0:
                        time_part_ratios.append(time.time())
                    mpi.world.barrier()
                else:
                    time_part_ratios.append(time.time())

                print("part ratio: {:0.2e}".format(part_ratio), end="\n")

    to_return = {
        "model": model,
        #"losses": torch.tensor(losses_train).detach(), #, losses_test]).detach(),
        "losses": torch.FloatTensor([losses_train, losses_test]).detach(),
        "models": models,
        "grad_norms": grad_norms,
        "lr": param_group["lr"],
        "time_epoch": time_epoch,
        "time_batch": time_batch,
    }

    if compute_part_ratio:
        to_return["part_ratios"] = torch.stack(part_ratios).detach()
        to_return["time_part_ratios"] = time_part_ratios

    return to_return