import matplotlib.pyplot as plt

def losses_plot(losses_train, losses_test, figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(losses_train, label='train')
    ax.plot(losses_test, label='test')
    ax.set_title('Losses')
    ax.set_ylabel('Losees')
    ax.set_xlabel('Iterations')
    return fig, ax

"""
def collective_variables_plot(C, R, figsize):
    
    return fig, ax

def acceptance_rate_plot():
    
    return fig, ax

def populations_convergence_plot():
    
    return fig, ax"""