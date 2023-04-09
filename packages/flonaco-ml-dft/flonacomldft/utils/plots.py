import torch
import numpy as np
import matplotlib.pyplot as plt
from flonacomldft.internal_coordinates import Coordinates_mapping

class analize_mlp_training:
    def __init__(self, mlp_training):
        self.mlp_training = mlp_training
    
    def plot_loss(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.mlp_training['losses'][0], label='loss train')
        ax.plot(self.mlp_training['losses'][1], label='loss test')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()

    def plot_correlation(self, datasets, ax=None):
        train, test = datasets
        
        coord_mapping = Coordinates_mapping()
        x_train, logdetjac_train, energy_train = coord_mapping.get_real_centered_from_internal(
                                    train[:, :12],
                                    train[:, 14],
                                    isomer=train[0, 13].int().item(),
                                    energies=train[:, 12]
                                    )

        x_test, logdetjac_test, energy_test = coord_mapping.get_real_centered_from_internal(
                                    test[:, :12],
                                    test[:, 14],
                                    isomer=test[0, 13].int().item(),
                                    energies=test[:, 12]
                                    )
        
        target_train = energy_train.detach().numpy()
        target_test = energy_test.detach().numpy()

        pred_train = self.mlp_training['model'](x_train).detach().numpy()
        pred_test = self.mlp_training['model'](x_test).detach().numpy()

        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(target_train, pred_train, label='train')
        ax.scatter(target_test, pred_test, label='test')
        min_ = pred_train.min()
        max_ = pred_train.max()
        range_ = np.linspace(min_, max_, 2)
        ax.plot(range_, range_, 'k--')
        ax.set_xlabel('Target energies')
        ax.set_ylabel('Predicted energies')
        ax.legend()


class analize_flow_training:
    def __init__(self, flow_training):
        self.flow_training = flow_training

    def plot_loss(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.flow_training['losses'][0], label='loss train')
        ax.plot(self.flow_training['losses'][1], label='loss test')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.legend()

    def plot_acc_ratio(self, ax=None, split=10):
        if ax is None:
            fig, ax = plt.subplots()
        for i, ratios in enumerate(self.flow_training['ratios']):
            acc_ratio = torch.stack(ratios['mlp']['acc_ratios']).mean(dim=1).detach().numpy()
            window_ratio = np.lib.stride_tricks.sliding_window_view(acc_ratio, split)
            ax.plot(np.arange(window_ratio.mean(-1).shape[0]), window_ratio.mean(-1), label='Model {:d}'.format(i))
            ax.plot(acc_ratio, alpha=0.1)
        ax.legend()
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Acceptance ratio')
        #ax.set_subtitle('Mode {:d} - Average Acceptance Ratio - Window {:d}'.format(mode_labels[0], split))

    def plot_part_ratio(self, ax=None, split=10):
        if ax is None:
            fig, ax = plt.subplots()
        for i, ratios in enumerate(self.flow_training['ratios']):
            part_ratio = torch.stack(ratios['mlp']['part_ratios']).detach().numpy()
            window_ratio = np.lib.stride_tricks.sliding_window_view(part_ratio, split)
            ax.plot(np.arange(window_ratio.mean(-1).shape[0]), window_ratio.mean(-1), label='Model {:d}'.format(i))
            ax.plot(part_ratio, alpha=0.1)
        ax.legend()
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Participation ratio')

    def plot_models_ratios(self, ax=None):
        first_acc_ratio = [torch.stack(ratios['mlp']['acc_ratios']).mean(dim=1).detach().numpy()[0] for ratios in self.flow_training['ratios']]
        last_acc_ratio = [torch.stack(ratios['mlp']['acc_ratios']).mean(dim=1).detach().numpy()[-1] for ratios in self.flow_training['ratios']]
        first_part_ratio = [torch.stack(ratios['mlp']['part_ratios']).detach().numpy()[0] for ratios in self.flow_training['ratios']]
        last_part_ratio = [torch.stack(ratios['mlp']['part_ratios']).detach().numpy()[-1] for ratios in self.flow_training['ratios']]

        ax.plot(first_acc_ratio, '-.o', label='First acc ratio')
        ax.plot(last_acc_ratio, '-.o', label='Last acc ratio')
        ax.plot(first_part_ratio, '-.o', label='First part ratio')
        ax.plot(last_part_ratio, '-.o', label='Last part ratio')
        ax.legend()
        ax.set_xlabel('Model')
        ax.set_ylabel('Ratio')


def plot_losses(losses_train, losses_test, title='losses', figsize=(8, 6), log_yscale=False):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(losses_train, label='train')
    ax.plot(losses_test, label='test')
    ax.set_title(title)
    ax.set_ylabel('Losses')
    ax.set_xlabel('Iterations')
    ax.legend()
    if log_yscale:
        ax.set_yscale('log')
    return fig, ax

def plot_sample(x, title, figsize=(8, 6)):

   from ase.visualize.plot import plot_atoms
   from flonacomldft.internal_coordinates import Coordinates_mapping
   
   fig, ax = plt.subplots(1, 1, figsize=figsize)
   ag6 = Coordinates_mapping()
   plot_atoms(ag6.get_molecule_from_internal(x), ax)
   ax.set_title(title)
   ax.set_xlabel('x coordinate')
   ax.set_ylabel('y coordinate')

   return fig, ax 

# TODO: add values as parameters to modify the FES plot

def plotting_fes_db():

   from flonacomldft.utils.io_utils import get_path
   from flonacomldft.FES.plotter2 import Plotter
   
   plotting = Plotter(400, 'Ag6')
   plotting.readfile(get_path() + 'unrotated_300.txt')
   
   fig, ax = plotting.plot_fes(0.1, 300, delta2=1, shift=1.5)
   
   return fig, ax

def collective_variables_plot(C, R, title, figsize=(8, 6)):
    
    fig, ax = plotting_fes_db()
    ax.scatter(C, R, color='orange')
    ax.set_title(title)

    return fig, ax

def acceptance_rate_plot(acceptance_rate, title, xlabel, figsize=(8, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(acceptance_rate)
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    return fig, ax

# TODO: fix this function
def populations_convergence_plot(populations, title, figsize=(8, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(populations)
    ax.set_title(title)
    ax.set_ylabel('Populations')
    ax.legend()
    return fig, ax

def plot_correlation_target_and_predict_value(target_train, predicted_train, target_test=None, predicted_test=None, title='', figsize=(6, 6)):
    
    import torch
    import numpy as np
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(target_train.detach().numpy(), predicted_train.detach().numpy()[:], label='train')
    if target_test is not None and predicted_test is not None:
        ax.scatter(target_test.detach().numpy(), predicted_test.detach().numpy()[:], label='test')
        predicted = torch.cat((predicted_train, predicted_test))
    else:
        predicted = predicted_train
    min_ = predicted.detach().numpy().min()
    max_ = predicted.detach().numpy().max()
    range_ = np.linspace(min_, max_, 2)
    ax.plot(range_, range_, 'k')
    ax.set_title(title)
    ax.set_ylabel('predicted value')
    ax.set_xlabel('actual value')
    ax.legend()
    
    return fig, ax