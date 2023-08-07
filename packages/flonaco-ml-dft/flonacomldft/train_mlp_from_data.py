### Import modules
import time
import copy
import tqdm
import torch
import torch.optim as optim

from ase.parallel import parprint as print

### MLP training function
def train_mlp(
    model,
    train,
    test,
    n_iter=100,
    lr=1e-4,
    use_scheduler=True,
    step_schedule=1000,
    save_splits=10,
    with_tqdm=False,
    dim=12,
    batch_size=512,
):
    """Train a MLP model on a dataset.

    Args:
        model: MLP model
        train: training dataset
        test: test dataset
        n_iter: number of iterations
        lr: learning rate
        use_scheduler: use scheduler
        step_schedule: step size for scheduler
        save_splits: number of splits to save
        with_tqdm: use tqdm

    Returns:
        model: trained MLP model
        losses: losses
        models: models
        grad_norms: gradient norms
    """
    
    ### MSE: Loss function with data centered
    def loss_func(x, y):
        ### x and y centered
        return ((model(x).squeeze() - y) ** 2).mean()

    ### Setting the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_schedule, gamma=0.5
        )

    ### Input and output data
    xs_train, us_train = train[:, :dim].clone(), train[:, dim].clone()
    xs_test, us_test = test[:, :dim].clone(), test[:, dim].clone() #(nsample_train, dims), (nsamples)
    
    if with_tqdm:
        pbar = tqdm.tqdm(range(n_iter))
    else:
        pbar = range(n_iter)


    print("train shape: ", xs_train.shape)
    print("test shape: ", xs_test.shape)

    permutation = torch.randperm(xs_train.shape[0])

    ### Logs
    losses_train = []
    losses_test = []
    grad_norms = []
    models = [copy.deepcopy(model)]

    ### Training loop
    for t in pbar:

        for k in range(0, xs_train.shape[0], batch_size):
            indexes = permutation[k:k+batch_size]
            batch_xs_train = xs_train[indexes].clone().detach().requires_grad_(True)
            batch_us_train = us_train[indexes].clone().detach().requires_grad_(True)

            optimizer.zero_grad()
            loss = loss_func(batch_xs_train, batch_us_train)
    
            if torch.isinf(loss).any():
                print("Stopped because loss became inf!")
                return model, losses_train
    
            ### Backpropagation
            loss.backward(retain_graph=True)
            optimizer.step()

        ### Losses
        losses_train.append(loss.item())
        losses_test.append(loss_func(xs_test, us_test).item())

        if with_tqdm:
            pbar.set_description(f"loss train: {losses_train[-1]:.4f}, loss test: {losses_test[-1]:.4f}")

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
        
        ### Save models and print logs
        if t % (n_iter / save_splits) == 0 or n_iter <= save_splits and with_tqdm==False:
            
            for param_group in optimizer.param_groups:
                lr = param_group["lr"]
            
            print("t={:0.1e} \t loss train: {:2.2e} \t loss test: {:2.2e} \t lr: {:.2e}".format(t, losses_train[-1], losses_test[-1], lr))
            
            models.append(copy.deepcopy(model))

    to_return = {
        "model": model,
        "train_loss": losses_train,
        "test_loss": losses_test,
        "models": models,
        "grad_norms": grad_norms,
    }

    return to_return