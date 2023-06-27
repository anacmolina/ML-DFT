### Import modules

import torch
import numpy as np
import matplotlib.pyplot as plt
from flonacomldft.internal_coordinates import Coordinates_mapping 
from flonacomldft.utils.io_utils import get_path
from flonacomldft.FES.plotter2 import Plotter
from flonacomldft.utils.diagnostics import avg_windows


### Define class to get the properties of a MD trajectory as a function of time

class MD_Properties:
    """Class to get the properties of a MD trajectory as a function of time."""
    def __init__(self):
        pass

    def get_total_energy(self, trajectory):
        """Get the total energy of the system as a function of time."""
        total_energy = torch.tensor([molecule.get_total_energy() for molecule in trajectory])
        total_energy = total_energy.detach().numpy()
        return total_energy
    
    def get_kinetic_energy(self, trajectory):
        """Get the kinetic energy of the system as a function of time."""
        kinetic_energy = torch.tensor([molecule.get_kinetic_energy() for molecule in trajectory])
        kinetic_energy = kinetic_energy.detach().numpy()
        return kinetic_energy
    
    def get_potential_energy(self, trajectory):
        """Get the potential energy of the system as a function of time."""
        potential_energy = torch.tensor([molecule.get_potential_energy() for molecule in trajectory])
        potential_energy = potential_energy.detach().numpy()
        return potential_energy
    
    def get_temperature(self, trajectory):
        """Get the temperature of the system as a function of time."""
        temperature = torch.tensor([molecule.get_temperature() for molecule in trajectory])
        temperature = temperature.detach().numpy()
        return temperature
    
    def get_collective_variables(self, trajectory):
        """Get the collective variables of the system as a function of time."""
        from flonacomldft.collective_variables import compute_C, compute_R
        cvs = np.array([[compute_C(molecule), compute_R(molecule)] for molecule in trajectory])
        return cvs

### Define class to plot the properties of a MD trajectory as a function of time

class MD_Plotter:

    """Class to plot the molecular dynamics of a system."""
    
    def __init__(self):
        pass
        
    def plot_total_energy(self, total_energy, ax=None):
        
        """Plot the energy of the system as a function of time."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(total_energy, label='Total Energy')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Energy (eV)')
        ax.legend()

    def plot_kinetic_energy(self, kinetic_energy, ax=None):
 
        """Plot the kinetic energy of the system as a function of time."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(kinetic_energy, label='Kinetic Energy')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Energy (eV)')
        ax.legend()   
        
    def plot_potential_energy(self, potential_energy, ax=None):

        """Plot the potential energy of the system as a function of time."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(potential_energy, label='Potential Energy')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Energy (eV)')
        ax.legend()
    
    def plot_temperature(self, temperature, ax=None, split=100):
        
        temp_avg = np.lib.stride_tricks.sliding_window_view(temperature, split).mean(axis=1)

        """Plot the temperature of the system as a function of time."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(temperature, alpha=0.8, label='Temperature')
        ax.plot(temp_avg, label='Temperature ({:d}-step average)'.format(split))
        ax.set_xlabel('Steps')
        ax.set_ylabel('Temperature (K)')
        ax.legend()


def set_plot_iteration(tensor, avg=True, window_size=10, axis=1, ax=None, init=0, color='k', alpha=0.5, label=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    x = init + np.arange(tensor.shape[0])
    y = tensor

    if avg:
        x_avg = np.arange(init + window_size - 1, init + window_size + len(avg_windows(tensor, window_size, axis)) - 1)
        y_avg = avg_windows(tensor, window_size, axis)
        ax.plot(x_avg, y_avg, color=color, label=label+'_avg', **kwargs)

    ax.plot(x, y, alpha=alpha, label=label, **kwargs)
    
    return ax

#TODO: Fix this function to add device and lims as optional arguments
lims = {'x_min':9.5, 'x_max':11., 'y_min':2.35, 'y_max':2.55}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_plot(ax, n_points, target_log_prob, prop_log_prob=None, lims=lims):
    
    x_range = torch.linspace(lims['x_min'], lims['x_max'], n_points, device=device)

    if lims['y_min'] is None:
        y_range = x_range.clone()
    else:
        y_range = torch.linspace(lims['y_min'], lims['y_max'], n_points, device=device)

    grid = torch.meshgrid(x_range, y_range)
    xys = torch.stack(grid).reshape(2, n_points ** 2).T.to(device)

    Us_target = target_log_prob(xys).reshape(n_points, n_points).T.detach().cpu().numpy()

    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)
    #plt.axis('off')

    plt.contourf(x_range, y_range, np.exp(Us_target)
    # np.log( - Us_target + Us_target.max() + 1)
    , 10, cmap='GnBu')
    ax.set_aspect('equal')

    if prop_log_prob is not None:
        Us_prop = prop_log_prob(xys).reshape(n_points, n_points).T.detach().cpu().numpy()

        plt.contour(x_range, y_range, np.exp(Us_prop), 10, colors='k', linestyles=':', alpha=0.5)


### Define class to plot the results of the Flonaco training and simulations

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


# TODO: DO AN OWN FES PLOTTER 

def plotting_fes_db():

   from flonacomldft.utils.io_utils import get_path
   from flonacomldft.FES.plotter2 import Plotter
   
   plotting = Plotter(400, 'Ag6')
   plotting.readfile(get_path() + 'unrotated_300.txt')
   
   fig, ax = plotting.plot_fes(0.1, 300, delta2=1, shift=1.5)

   return fig, ax

#===================================================================================================

#TODO: Plots to add to Flonaco_Plotter

def plot_acc_ratio(self, ax=None, split=10):
    """Plot the average acceptance ratio of the normalizing flow as a function of the number of iterations."""
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
    """Plot the average participation ratio of the normalizing flow as a function of the number of iterations."""
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
    """Plot the average acceptance ratio and participation ratio of the normalizing flow as a function of the number of iterations."""
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
 
def acceptance_rate_plot(acceptance_rate, title, xlabel, figsize=(8, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(acceptance_rate)
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    return fig, ax

def populations_convergence_plot(populations, title, figsize=(8, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(populations)
    ax.set_title(title)
    ax.set_ylabel('Populations')
    ax.legend()
    return fig, ax