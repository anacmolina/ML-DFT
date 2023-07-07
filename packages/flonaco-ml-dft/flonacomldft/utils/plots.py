# libraries

import torch
import numpy as np

from flonacomldft.utils.io_utils import get_path
from flonacomldft.utils.diagnostics import avg_windows
from flonacomldft.internal_coordinates import Coordinates_mapping 

import matplotlib.pyplot as plt
import seaborn as sns
import cmocean.cm as cmo

from ase.units import kB
from flonacomldft.FES.plotter2 import Plotter


def set_plot_sequential_data(tensor, avg=True, window_size=10, axis=1, ax=None, init=0, color='k', alpha=0.5, label=None, **kwargs):
    """
    Function to plot sequential data.
    Args:
        tensor (torch.Tensor): Tensor to plot.
        avg (bool): Whether to plot the average of the tensor.
        window_size (int): Size of the window to average the tensor.
        axis (int): Axis to average the tensor.
        ax (matplotlib.axes.Axes): Axes to plot the tensor.
        init (int): Initial value of the x-axis.
        color (str): Color of the plot.
        alpha (float): Transparency of the plot.
        label (str): Label of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    x = init + np.arange(tensor.shape[0])
    y = tensor

    if avg:
        x_avg = np.arange(init + window_size - 1, 
                          init + window_size + len(avg_windows(tensor, window_size, axis)) - 1)
        y_avg = avg_windows(tensor, window_size, axis)

        ax.plot(x_avg, y_avg, color=color, **kwargs)

    ax.plot(x, y, alpha=alpha, label=label, **kwargs)

    return ax


def plot_conformations(conformations, cols=5, inds=None, figsize=(15, 5)):
    """Plot a list of conformations.
    Args:
        conformations (list): List of ase.Atoms objects.
        cols (int): Number of columns of the plot.
        figsize (tuple): Size of the figure.
    Returns:
        fig (matplotlib.figure.Figure): Figure of the plot.
        axs (list): List of matplotlib.axes.Axes objects.
    """
    
    from ase.visualize.plot import plot_atoms

    N = len(conformations)
    rows = int(N/cols)

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.ravel()
    
    for i, ax in zip(range(N), axs):
        plot_atoms(conformations[i], ax)
        if inds is not None:
            ax.set_title('Conformation {:d}'.format(inds[i]))
        else:
            ax.set_title('Conformation {:d}'.format(i))

    return fig, axs


def plot_energy_histogram(energy_molecules, axis_labels='default', ax=None, **kwargs):
    """ Plot the histogram of the energies of a list of molecules.
    Args:
        energy_molecules (dataframe): Dataframe with the energies of the molecules.
        axis_labels (str): Whether to plot the default axis labels.
        ax (matplotlib.axes.Axes): Axes to plot the histogram.
        **kwargs: Keyword arguments to pass to the seaborn.histplot function.
    Returns:
        ax (matplotlib.axes.Axes): Axes of the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    sns.histplot(energy_molecules, stat='density', **kwargs)

    if axis_labels == 'default':
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Density (u.a.)')

    return ax

def plot_energy_surface(C=None, R=None, F=None, bins=401, shift=2.0, delta=4 * (kB * 300), T=300, fig=None, ax=None, cmap=cmo.tempo_r, add_colorbar=True, axis_labels='default'):
    """ Plot the free energy surface 
    Args:
        C (np.array): Array with the values of the C coordinate.
        R (np.array): Array with the values of the R coordinate.
        F (np.array): Array with the values of the free energy.
        bins (int): Number of bins to plot the free energy surface.
        shift (float): Shift to apply to the free energy.
        delta (float): Isoline spacing.
        T (float): Temperature of the simulation.
        fig (matplotlib.figure.Figure): Figure of the plot.
        ax (matplotlib.axes.Axes): Axes to plot the free energy surface.
        cmap (matplotlib.colors.Colormap): Colormap to use.
        add_colorbar (bool): Whether to add a colorbar to the plot.
        axis_labels (str): Whether to plot the default axis labels.
    Returns:
        fig (matplotlib.figure.Figure): Figure of the plot.
        ax (matplotlib.axes.Axes): Axes of the plot.    
    """
    
    if C is None or R is None or F is None:
        data = np.loadtxt(get_path() + '/unrotated_300.txt')
        C, R, F = data.T

    C_grid = C.reshape(bins, bins)
    R_grid = R.reshape(bins, bins)

    shift = 2.0
    delta = 4 * (kB * T)

    F = F + shift
    F[~(F<=0)] = None

    mini, maxi = F[~np.isnan(F)].min(), F[~np.isnan(F)].max()

    F_grid = F.reshape(bins, bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.contourf(C_grid, R_grid, F_grid, np.arange(mini, maxi, delta), cmap=cmap, vmin=mini, vmax=maxi)

    if fig is not None and add_colorbar:
        fig.colorbar(im, label = 'Free energy (eV)')
    elif fig is None:
        print('No figure provided. Cannot add colorbar.')

    if axis_labels=='default':
        ax.set_xlabel('Coordination number')
        ax.set_ylabel('Radius of gyration (Å)')

    if fig is not None:
        return fig, ax
    else:
        return ax


     
class MD_Plotter:

    """Class to plot the molecular dynamics of a system."""
    
    def __init__(self, traj_properties):
        self.traj_properties = traj_properties

        
    def plot_total_energy(self, avg=False, window_size=10, axis_labels='default', **kwargs):

        """Plot the total energy of the system as a function of time."""

        ax = set_plot_sequential_data(self.traj_properties.total_energy, avg=avg, window_size=window_size, **kwargs)
        
        if axis_labels == 'default':
            ax.set_xlabel('Steps')
            ax.set_ylabel('Energy (eV)')
            ax.legend()
        else:
            pass

    def plot_kinetic_energy(self, avg=False, window_size=10, axis_labels='default', **kwargs):
 
        """Plot the kinetic energy of the system as a function of time."""
        
        ax = set_plot_sequential_data(self.traj_properties.kinetic_energy, avg=avg, window_size=window_size, **kwargs)

        if axis_labels == 'default':
            ax.set_xlabel('Steps')
            ax.set_ylabel('Energy (eV)')
            ax.legend()   
        
    def plot_potential_energy(self, avg=False, window_size=10, axis_labels='default', **kwargs):

        """Plot the potential energy of the system as a function of time."""
        
        ax = set_plot_sequential_data(self.traj_properties.potential_energy, avg=avg, window_size=window_size, **kwargs)

        if axis_labels == 'default':
            ax.set_xlabel('Steps')
            ax.set_ylabel('Energy (eV)')
            ax.legend()
    
    def plot_temperature(self, avg=False, window_size=10, axis_labels='default', **kwargs):

        """Plot the temperature of the system as a function of time."""

        ax = set_plot_sequential_data(self.traj_properties.temperature, avg=avg, window_size=window_size, **kwargs)

        if axis_labels == 'default':
            ax.set_xlabel('Steps')
            ax.set_ylabel('Temperature (K)')
            ax.legend()

    def plot_trajectory_conformations(self, inds=None, cols=5, figsize=(15, 5)):
        
        """Plot the conformations of the trajectory."""
        
        selected_conformations = [self.traj_properties.trajectory[i] for i in inds]
        print(inds)

        plot_conformations(selected_conformations, inds=inds, cols=cols, figsize=figsize)    


    # TODO Add collective variables plots: time, 2d probability and point on fes                              


class NF_Plotter:

    def __init__(self, flow_dict, isomer=None):
        self.flow_dict = flow_dict

    def plot_losses(self, train_loss, test_loss, avg=False, window_size=50, axis_labels=True, **kwargs):

        ax = set_plot_sequential_data(train_loss, label='Train loss', avg=avg, window_size=window_size, **kwargs)
        ax = set_plot_sequential_data(test_loss, label='Test loss', avg=avg, window_size=window_size, **kwargs, ax=ax)

        if axis_labels:
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Loss')
            ax.legend()

        return ax
    
    def plot_part_ratio(self, part_ratio, avg=False, window_size=50, axis_labels=True, **kwargs):

        ax = set_plot_sequential_data(part_ratio, label='Partition ratio', avg=avg, window_size=window_size, **kwargs)

        if axis_labels:
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Participation ratio')

        return ax
    
    def plot_flow_samples(self, samples, **kwargs):

        plot_conformations(samples, **kwargs)

    def plot_collective_variables(self, ax, color='r', marker_size=10, label='cvs', model_number=-1, n_samples=10, **kwargs):

        if self.flow_dict.collective_variables is not None:

            ax.scatter(self.flow_dict.collective_variables[:, 0], 
                   self.flow_dict.collective_variables[:, 1], 
                   s=marker_size, 
                   c=color, 
                   label=label)        

        else:
            print('Collective variables not computed. Cannot plot them.')
        

        

#================================= REVIEW ====================================

#TODO: Fix this function to add device and lims as optional arguments

#lims = {'x_min':9.5, 'x_max':11., 'y_min':2.35, 'y_max':2.55}
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: This for the FES
def set_plot_probability(n_points, lims, target_log_prob, prop_log_prob=None, ax=None, device='cpu'):
    """
    Function to plot the probability of a target distribution and a proposal distribution if provided.
    Args:
        n_points (int): Number of grid points to plot.
        lims (dict): Dictionary with the limits of the plot.
        target_log_prob (callable): Target log probability function.
        prop_log_prob (callable): Proposal log probability function.
        ax (matplotlib.axes.Axes): Axes to plot the probability.
        device (str): Device to compute the probability.
    """

    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)

    x_range = torch.linspace(lims['x_min'], lims['x_max'], n_points, device=device)

    if lims['y_min'] is None:
        y_range = x_range.clone()
    else:
        y_range = torch.linspace(lims['y_min'], lims['y_max'], n_points, device=device)

    grid = torch.meshgrid(x_range, y_range)
    xys = torch.stack(grid).reshape(2, n_points ** 2).T.to(device)

    us_target = target_log_prob(xys).reshape(n_points, n_points).T.detach().cpu().numpy()

    plt.contourf(x_range, y_range, np.exp(us_target), 10, cmap='GnBu')
    ax.set_aspect('auto')

    if prop_log_prob is not None:
        
        us_prop = prop_log_prob(xys).reshape(n_points, n_points).T.detach().cpu().numpy()

        plt.contour(x_range, y_range, np.exp(us_prop), 10, colors='k', linestyles=':', alpha=0.5)

# ================================== DELETE ====================================

#TODO: DELETE THIS FUNCTION
#class for plotling flow, metropolis and adaptive results
class Flonaco_Plotter:
    
    """Class to plot the results of the Flonaco training and simulations."""
    
    def __init__(self):
        pass

    def plot_losses(self, losses, yscale=True, ax=None):
        
        """Plot the train and test loss of a DL model."""
        
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.plot(losses[0], label='loss train')
        ax.plot(losses[1], label='loss test')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        
        if yscale:
            ax.set_yscale('log')
        
        ax.legend()

    def plot_correlation(self, model, datasets, ax=None):
        
        """Plot the correlation between the predicted and the true energy of the molecules."""
        
        train, test = datasets
        
        x_train = train[:, :12]
        x_test = test[:, :12]

        target_train = train[:, 12].detach().numpy()
        target_test = test[:, 12].detach().numpy()

        pred_train = model(x_train).detach().numpy()
        pred_test = model(x_test).detach().numpy()

        if ax is None:
            fig, ax = plt.subplots()
            
        ax.scatter(target_train, pred_train, label='train')
        ax.scatter(target_test, pred_test, label='test')

        min_ = target_train.min()
        max_ = target_train.max()
        range_ = np.linspace(min_, max_, 2)
        ax.plot(range_, range_, 'k--')
        ax.set_xlabel('Target energies')
        ax.set_ylabel('Predicted energies')
        ax.legend()
    
    def plot_collective_variables_on_time(self, cvs, ax=None):

        """Plot the collective variables of the system as a function of time."""

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(cvs[:, 0], label='Coordination Number')
        ax.plot(cvs[:, 1], label='Radius of Gyration (Å)')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Collective Variables')
        ax.legend()
    
    def plot_collective_variables_on_fes(self, cvs, label, marker='o', cmap='autumn'):

        """ Plot the collective variables of the system on the free energy surface."""

        plotter = Plotter(400, 'Ag6', )
        plotter.readfile(get_path() + '/' + 'unrotated_300.txt')

        fig, ax = plotter.plot_fes(0.1, 300, delta2=1, shift=1.5)
        
        ax.scatter(cvs[:, 0], cvs[:, 1], marker=marker, cmap=cmap, label=label)
        fig.set_size_inches(10, 6)
        size = 12
        ax.xaxis.label.set_size(size)
        ax.yaxis.label.set_size(size)
        plt.xticks(fontsize=size)
        plt.yticks(fontsize=size)
        ax.legend(fontsize=size);
    
    def plot_sample(self, zmat, title, ax=None, figsize=(8, 6)):

        """Plot the molecule from the internal coordinates."""

        from ase.visualize.plot import plot_atoms

        ag6 = Coordinates_mapping()
        molecule = ag6.build_molecule_from_zmat(zmat)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
        plot_atoms(molecule, ax)
        ax.set_title(title)
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')

    def plot_internal_coordinates(self, zmats, ax=None):
    
        """Plot range value of internal coordinates."""

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.plot(zmats.T, 'o')
        ax.set_xlabel('Internal coordinates')
        ax.set_ylabel('a. u')
        
# TODO: DELETE!!!!!!!!!!!
def plotting_fes_db():

   from flonacomldft.utils.io_utils import get_path
   from flonacomldft.FES.plotter2 import Plotter
   
   plotting = Plotter(400, 'Ag6')
   plotting.readfile(get_path() + 'unrotated_300.txt')
   
   fig, ax = plotting.plot_fes(0.1, 300, delta2=1, shift=1.5)

   return fig, ax

# 
# #===================================================================================================
# 
# def plot_acc_ratio(self, ax=None, split=10):
#     """Plot the average acceptance ratio of the normalizing flow as a function of the number of iterations."""
#     if ax is None:
#         fig, ax = plt.subplots()
#     for i, ratios in enumerate(self.flow_training['ratios']):
#         acc_ratio = torch.stack(ratios['mlp']['acc_ratios']).mean(dim=1).detach().numpy()
#         window_ratio = np.lib.stride_tricks.sliding_window_view(acc_ratio, split)
#         ax.plot(np.arange(window_ratio.mean(-1).shape[0]), window_ratio.mean(-1), label='Model {:d}'.format(i))
#         ax.plot(acc_ratio, alpha=0.1)
#     ax.legend()
#     ax.set_xlabel('Iterations')
#     ax.set_ylabel('Acceptance ratio')
#     #ax.set_subtitle('Mode {:d} - Average Acceptance Ratio - Window {:d}'.format(mode_labels[0], split))
#  
# def plot_part_ratio(self, ax=None, split=10):
#     """Plot the average participation ratio of the normalizing flow as a function of the number of iterations."""
#     if ax is None:
#         fig, ax = plt.subplots()
#     for i, ratios in enumerate(self.flow_training['ratios']):
#         part_ratio = torch.stack(ratios['mlp']['part_ratios']).detach().numpy()
#         window_ratio = np.lib.stride_tricks.sliding_window_view(part_ratio, split)
#         ax.plot(np.arange(window_ratio.mean(-1).shape[0]), window_ratio.mean(-1), label='Model {:d}'.format(i))
#         ax.plot(part_ratio, alpha=0.1)
#     ax.legend()
#     ax.set_xlabel('Iterations')
#     ax.set_ylabel('Participation ratio')
#                   
# def plot_models_ratios(self, ax=None):
#     """Plot the average acceptance ratio and participation ratio of the normalizing flow as a function of the number of iterations."""
#     first_acc_ratio = [torch.stack(ratios['mlp']['acc_ratios']).mean(dim=1).detach().numpy()[0] for ratios in self.flow_training['ratios']]
#     last_acc_ratio = [torch.stack(ratios['mlp']['acc_ratios']).mean(dim=1).detach().numpy()[-1] for ratios in self.flow_training['ratios']]
#     first_part_ratio = [torch.stack(ratios['mlp']['part_ratios']).detach().numpy()[0] for ratios in self.flow_training['ratios']]
#     last_part_ratio = [torch.stack(ratios['mlp']['part_ratios']).detach().numpy()[-1] for ratios in self.flow_training['ratios']]
#     ax.plot(first_acc_ratio, '-.o', label='First acc ratio')
#     ax.plot(last_acc_ratio, '-.o', label='Last acc ratio')
#     ax.plot(first_part_ratio, '-.o', label='First part ratio')
#     ax.plot(last_part_ratio, '-.o', label='Last part ratio')
#     ax.legend()
#     ax.set_xlabel('Model')
#     ax.set_ylabel('Ratio')
#  
# def acceptance_rate_plot(acceptance_rate, title, xlabel, figsize=(8, 6)):
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     ax.plot(acceptance_rate)
#     ax.set_title(title)
#     ax.set_ylabel(title)
#     ax.set_xlabel(xlabel)
#     ax.legend()
#     return fig, ax
# 
# def populations_convergence_plot(populations, title, figsize=(8, 6)):
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     ax.plot(populations)
#     ax.set_title(title)
#     ax.set_ylabel('Populations')
#     ax.legend()
#     return fig, ax