### Import modules
import torch
import numpy as np
import matplotlib.pyplot as plt
from flonacomldft.internal_coordinates import Coordinates_mapping
from flonacomldft.collective_variables import get_CVs
from flonacomldft.utils.io_utils import get_path
from flonacomldft.FES.plotter2 import Plotter

### Define class to plot molecular dynamics results
class plot_molecular_dynamics:
    """Class to plot the molecular dynamics of a system."""
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
    
    def plot_collective_variables_versus_time(self, cvs, ax=None):

        """Plot the collective variables of the system as a function of time."""

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(cvs[:, 0], label='Coordination Number')
        ax.plot(cvs[:, 1], label='Radius of Gyration (Å)')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Collective Variables')
        ax.legend()
    
    def plot_collective_variables_on_fes(self, cvs, name, cmap='autumn'):

        """ Plot the collective variables of the system on the free energy surface."""

        plotter = Plotter(400, 'Ag6', )
        plotter.readfile(get_path() + 'unrotated_300.txt')

        fig, ax = plotter.plot_fes(0.1, 300, delta2=1, shift=1.5)
        
        ax.scatter(cvs[:, 0], cvs[:, 1], marker='x', cmap=cmap, label='MD {:s}'.format(name))
        fig.set_size_inches(10, 6)
        size = 12
        ax.xaxis.label.set_size(size)
        ax.yaxis.label.set_size(size)
        plt.xticks(fontsize=size)
        plt.yticks(fontsize=size)
        ax.legend(fontsize=size);


### Define function to plot the internal coordinates of a molecule    
def plot_internal_coordinates(zmat, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(zmat.T, 'o')
    ax.set_xlabel('Internal coordinates')
    ax.set_ylabel('a. u')

### Define class to plot the training of a neural network
class plot_mlp_training:
    """Class to plot the training of a neural network to predict the energy of a molecule."""

    def __init__(self, mlp_training):
        self.mlp_training = mlp_training
    
    def plot_loss(self, ax=None):
        """Plot the loss of the neural network as a function of the number of iterations."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.mlp_training['losses'][0], label='loss train')
        ax.plot(self.mlp_training['losses'][1], label='loss test')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()

    def plot_correlation(self, datasets, ax=None):
        """Plot the correlation between the predicted and the true energy of the molecules."""
        train, test = datasets ### real centered coordinates
        
        x_train = train[:, :12]
        x_test = test[:, :12]

        target_train = train[:, 12].detach().numpy()
        target_test = test[:, 12].detach().numpy()

        pred_train = self.mlp_training['model'](x_train).detach().numpy()
        pred_test = self.mlp_training['model'](x_test).detach().numpy()

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


class plot_flow_training:
    
    """Class to plot the training of a normalizing flow to propose new molecules."""
    
    def __init__(self, flow_training):
        self.flow_training = flow_training

    def plot_loss(self, ax=None):
        """Plot the loss of the normalizing flow as a function of the number of iterations."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.flow_training['losses'][0], label='loss train')
        ax.plot(self.flow_training['losses'][1], label='loss test')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.legend()

    def plot_samples_on_fes(self, model, n_samples, isomer, cmap='autumn'):
        xs = model.sample(n_samples)

        coord_mapping = Coordinates_mapping()
        zmat, logdetjacs_zmat = coord_mapping.get_internal_from_real_centered(xs, isomer=isomer)
        
        cvs = get_CVs(zmat)
        cvs = np.array(cvs)
        
        plotter = Plotter(400, 'Ag6', )
        plotter.readfile(get_path() + 'unrotated_300.txt')
        
        fig, ax = plotter.plot_fes(0.1, 300, delta2=1, shift=1.5)
        
        ax.scatter(cvs[0], cvs[1], marker='x', cmap=cmap, label='Samples is{:d}'.format(isomer))
        fig.set_size_inches(10, 6)
        size = 12
        ax.xaxis.label.set_size(size)
        ax.yaxis.label.set_size(size)
        plt.xticks(fontsize=size)
        plt.yticks(fontsize=size)
        ax.legend(fontsize=size);
        
        return fig, ax

#    def plot_acc_ratio(self, ax=None, split=10):
#        """Plot the average acceptance ratio of the normalizing flow as a function of the number of iterations."""
#        if ax is None:
#            fig, ax = plt.subplots()
#        for i, ratios in enumerate(self.flow_training['ratios']):
#            acc_ratio = torch.stack(ratios['mlp']['acc_ratios']).mean(dim=1).detach().numpy()
#            window_ratio = np.lib.stride_tricks.sliding_window_view(acc_ratio, split)
#            ax.plot(np.arange(window_ratio.mean(-1).shape[0]), window_ratio.mean(-1), label='Model {:d}'.format(i))
#            ax.plot(acc_ratio, alpha=0.1)
#        ax.legend()
#        ax.set_xlabel('Iterations')
#        ax.set_ylabel('Acceptance ratio')
#        #ax.set_subtitle('Mode {:d} - Average Acceptance Ratio - Window {:d}'.format(mode_labels[0], split))
 
#    def plot_part_ratio(self, ax=None, split=10):
#        """Plot the average participation ratio of the normalizing flow as a function of the number of iterations."""
#        if ax is None:
#            fig, ax = plt.subplots()
#        for i, ratios in enumerate(self.flow_training['ratios']):
#            part_ratio = torch.stack(ratios['mlp']['part_ratios']).detach().numpy()
#            window_ratio = np.lib.stride_tricks.sliding_window_view(part_ratio, split)
#            ax.plot(np.arange(window_ratio.mean(-1).shape[0]), window_ratio.mean(-1), label='Model {:d}'.format(i))
#            ax.plot(part_ratio, alpha=0.1)
#        ax.legend()
#        ax.set_xlabel('Iterations')
#        ax.set_ylabel('Participation ratio')

#    def plot_models_ratios(self, ax=None):
#        """Plot the average acceptance ratio and participation ratio of the normalizing flow as a function of the number of iterations."""
#        first_acc_ratio = [torch.stack(ratios['mlp']['acc_ratios']).mean(dim=1).detach().numpy()[0] for ratios in self.flow_training['ratios']]
#        last_acc_ratio = [torch.stack(ratios['mlp']['acc_ratios']).mean(dim=1).detach().numpy()[-1] for ratios in self.flow_training['ratios']]
#        first_part_ratio = [torch.stack(ratios['mlp']['part_ratios']).detach().numpy()[0] for ratios in self.flow_training['ratios']]
#        last_part_ratio = [torch.stack(ratios['mlp']['part_ratios']).detach().numpy()[-1] for ratios in self.flow_training['ratios']]
#
#        ax.plot(first_acc_ratio, '-.o', label='First acc ratio')
#        ax.plot(last_acc_ratio, '-.o', label='Last acc ratio')
#        ax.plot(first_part_ratio, '-.o', label='First part ratio')
#        ax.plot(last_part_ratio, '-.o', label='Last part ratio')
#        ax.legend()
#        ax.set_xlabel('Model')
#        ax.set_ylabel('Ratio')


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
   plot_atoms(ag6.build_molecule_from_internal(x), ax)
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