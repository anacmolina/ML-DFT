### Import modules
import time
import copy
import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
# from ray import tune

from ase.parallel import parprint as print

### Flow training function
def train_flow(
    model,
    train,
    test,
    dim=12,
    n_iter=100,
    lr=5e-3,
    batch_size=128,
    use_scheduler=False,
    step_schedule=1000,
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
        test: test dataset
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
    

    ### Setting up the Loss function with data centered
    def loss_func(x):
        return (model.nll(x)).mean() ### negative log likelihood

    ### Setting the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_schedule, gamma=0.5
        )

    ### Input data   
    x_train = train[:, :dim]
    x_test = test[:, :dim]

    if compute_part_ratio:
        from flonacomldft.utils.diagnostics import Target_Log_Prob
        from flonacomldft.utils.diagnostics import get_participation_ratio

        mode_label = train[:, dim+1].unique().int()
        
        energy_type = energy_type
        mlp_model = mlp_model

        part_ratios = []

    ### Logs
    losses_train = []
    losses_test = []
    models = [copy.deepcopy(model)]
    grad_norms = []

    if x_train.shape[0] < 8000:

        x = x_train.detach().requires_grad_()

    else:

        x = x_train.detach().requires_grad_()[-8000:]

    if with_tqdm:
        pbar = tqdm.tqdm(range(n_iter))
    else:
        pbar = range(n_iter)

    # num_model = 0   

    time_start = time.time()

    print("train shape: ", x.shape)

    permutation = torch.randperm(x.shape[0])

    ### Training loop
    for t in pbar:

        for  k in range(0, x.shape[0], batch_size):
            indexes = permutation[k:k+batch_size]
            batch_x = x[indexes]

            optimizer.zero_grad()
            loss = loss_func(batch_x)

            if torch.isinf(loss).any():
                print("Stopped because loss became inf!")
                return model, losses_train

            ### Backpropagation
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        ### Appending logs
        losses_train.append(loss.item())
        losses_test.append(loss_func(x_test).item())

        if with_tqdm:
            pbar.set_description(f"Loss: {losses_train[-1]:.4f}")
        
        ### Gradient norms
        if t % (n_iter / 100) == 0:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)

        ### Learning rate scheduler
        if use_scheduler:
            scheduler.step()

        ### Saving models and printing logs
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
                    path = path+'/DFTComputations_{:d}'.format(t)
                else:
                    path = None

                
                ##TODO: Check if this is necessary
                #import gpaw.mpi as mpi
                    
                target_log_prob = Target_Log_Prob(energy_type=energy_type, mode_label=mode_label, mlp_model=mlp_model, folder=path).target_log_prob
                part_ratio = get_participation_ratio(model, target_log_prob, n_prop=n_prop)

                #if mpi.rank == 0:
                #    num_model += 1

                #mpi.world.barrier()

                part_ratios.append(part_ratio)

                print("part ratio: {:0.2e}".format(part_ratio), end="\n")

    to_return = {
        "model": model,
        "losses": torch.FloatTensor([losses_train, losses_test]).detach(),
        "models": models,
        "grad_norms": grad_norms,
    }

    if compute_part_ratio:
        to_return["part_ratios"] = torch.stack(part_ratios).detach()

    return to_return