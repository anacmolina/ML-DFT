import matplotlib.pyplot as plt

def losses_plot(losses_train, losses_test, title, figsize=(8, 6), log_yscale=False):
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
   from flonacomldft.internal_coordinates import Structure
   
   fig, ax = plt.subplots(1, 1, figsize=figsize)
   ag6 = Structure()
   plot_atoms(ag6.build_molecule(x), ax)
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
    ax.set_ylabel('Acceptance rate')
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
    ax.scatter(target_train, predicted_train.detach().numpy()[:], label='train')
    if target_test is not None and predicted_test is not None:
        ax.scatter(target_test, predicted_test.detach().numpy()[:], label='test')
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