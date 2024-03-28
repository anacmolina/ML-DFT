### Import modules
import time
import copy
import tqdm
import torch
from torch.nn.utils import clip_grad_norm_

from ase.parallel import parprint as print

#TODO: add docstring
#TODO: add device

def train_mlp(
    model,
    train,
    test,
    n_iter,
    lr,
    bs,
    use_scheduler=False,
    step_scheduler=10,
    save_splits=1,
    grad_clip=1e4,
    with_tqdm=False,
    n_partial_loss=10,
    dim=12,  
    ):
    """
    Train a MLP model using the mean squared error as loss function.
    Args:
        model (): the model to train
        train (torch.Tensor): the training data
        test (torch.Tensor): the test data
        n_iter (int): the number of iterations
        lr (float): the learning rate
        bs (int): the batch size
        use_scheduler (bool): whether to use a scheduler
        step_scheduler (int): the step size for the scheduler
        save_splits (int): the number of models to save
        grad_clip (float): the maximum gradient norm
        with_tqdm (bool): whether to use tqdm
        n_partial_loss (int): the number of partial losses to print
        dim (int): the dimension of the input data
    """

    
    def loss_func(x, y):
        return ((model(x).squeeze() - y)**2).mean()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=step_scheduler, 
                                                    gamma=0.5)

    if bs <= train.shape[0]:
        n_epochs = round( n_iter * bs / train.shape[0] )
    else:
        n_epochs = n_iter

    if n_epochs < 1:
        n_epochs = 1

    print('Number of epochs: ', n_epochs)

    if save_splits > 1:
        models = [copy.deepcopy(model)]

    if with_tqdm:
        pbar = tqdm.tqdm(range(n_epochs))
    else:
        pbar = range(n_epochs)
        print('Epoch \t\t Lr \t\t Train Loss \t Test Loss \t Grad norm')


    train_losses = []
    test_losses = []
    avg_train_losses = []
    avg_test_losses = []
    grad_norms = []
    time_step = []

    x_train, y_train = train[:, :dim], train[:, dim]
    x_test, y_test = test[:, :dim], test[:, dim]

    permutation = torch.randperm(x_train.size()[0])

    for t in pbar:

        avg_train_loss = 0
        avg_test_loss = 0

        for k in range(0, x_train.size()[0], bs):

            indices = permutation[k:k+bs]
            x_batch, y_batch = x_train[indices].detach(), y_train[indices].detach()

            optimizer.zero_grad()
            loss = loss_func(x_batch, y_batch)
            
            if torch.isinf(loss).any():
                print('Stopped because loss became of inf!')
                return {'model': model, 
                        'train_loss': train_losses, 
                        'test_losses': test_losses}

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            train_losses.append(loss.item())
            test_losses.append(loss_func(x_test, y_test).item())

            avg_train_loss += loss.item()
            avg_test_loss += loss_func(x_test, y_test).item()

        avg_train_loss /= (x_train.size()[0] / bs)
        avg_test_loss /= (x_test.size()[0] / bs)

        avg_train_losses.append(avg_train_loss)
        avg_test_losses.append(avg_test_loss)

        time_step.append(time.time())

        if t % (n_epochs / 100) == 0:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)

        if use_scheduler:
            scheduler.step()

        if with_tqdm == False:

            if n_partial_loss > n_epochs:
                n_partial_loss = n_epochs

            if t % (n_epochs // n_partial_loss) == 0:

                for param_group in optimizer.param_groups:
                    lr_ = param_group['lr']

                print('{:0.1e} \t {:0.2e} \t {:3.2e} \t {:3.2e} \t {:0.1e}'.format(t, lr_, avg_train_losses[-1], avg_test_losses[-1], grad_norms[-1]))

        else:

            pbar.set_description('Loss: {:.4f}'.format(train_losses[-1]))


        if save_splits > 1:
            if t % (n_epochs // save_splits) == 0:
                models.append(copy.deepcopy(model))

    to_return = {'model': model,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'avg_train_losses': avg_train_losses,
                'avg_test_losses': avg_test_losses,
                'grad_norms': grad_norms,
                'time_step': time_step}
    
    if save_splits > 1:
        to_return['models'] = models

    return to_return