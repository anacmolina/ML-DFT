# libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import cmocean.cm as cmo
from ase.units import kB

from abflowmc.utils.io_utils import get_path
from abflowmc.utils.io_utils import get_project_path

load_ag6_image = lambda i: np.asarray(Image.open(get_project_path() + '/images/ag6_is{:d}.png'.format(i)))

def avg_windows(x, window_size, axis):
    return np.lib.stride_tricks.sliding_window_view(x, window_size).mean(axis=axis)

def set_plot_sequential_data(y, x=None, avg=True, window_size=10, axis=1, ax=None, init=0, color='k', alpha=0.3, label=None, **kwargs):
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
    
    if x is None:
        x = init + np.arange(y.shape[0])

    if avg:
        if x is None:
            x_avg = np.arange(init + window_size - 1, 
                              init + window_size + len(avg_windows(y, window_size, axis)) - 1)
        else:
            x_avg = avg_windows(x, window_size, axis)
        y_avg = avg_windows(y, window_size, axis)


        ax.plot(x_avg, y_avg, color=color, label=label, **kwargs)

    ax.plot(x, y, alpha=alpha, color=color, **kwargs)

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


def plot_energy_histogram(energy_molecules, axis_labels='default', stat='density', common_bins=False, common_norm=False, ax=None, **kwargs):
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

    sns.histplot(energy_molecules, stat=stat, ax=ax, common_bins=common_bins, common_norm=common_norm, **kwargs)

    if axis_labels == 'default':
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Density (u.a.)')

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
        

def create_report(plot_data):

    fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    fig.subplots_adjust(top=0.92)
    axs[0][0].scatter(plot_data['cvs'][:, 0], plot_data['cvs'][:, 1], c='orange')
    axs[0][0].set_xlabel("Coordination number")
    axs[0][0].set_ylabel("Radius of gyration")

    axs[0][1].scatter(plot_data['energies']['adaptive'], 
                      plot_data['nlls'], 
                      c=np.arange(len(plot_data['nlls'])), 
                      cmap='viridis')
    
    axs[0][1].set_xlabel("Energy")
    axs[0][1].set_ylabel("NLL")

    set_plot_sequential_data(plot_data['accs'], ax=axs[1][0])
    axs[1][0].set_xlabel("MCMC steps")
    axs[1][0].set_ylabel("Acceptance ratio")

    axs[1][1].plot(plot_data['part_ratios'], 'o--')
    axs[1][1].set_xlabel("MCMC steps")
    axs[1][1].set_ylabel("Participation ratio")

    set_plot_sequential_data(plot_data['losses'], ax=axs[2][0])
    axs[2][0].set_xlabel("Iterations")
    axs[2][0].set_ylabel("Loss")

    plot_energy_histogram(plot_data['energies'], ax=axs[2][1])

    return fig, axs
