import torch
from ase.units import kB

#TODO: APPARENTLY THIS IS NOT USED ANYMORE

def run_mcmc(energies, steps, Nchains):
    
    # Shuffle all the configurations
    samples = torch.randperm(len(trajectory))
    
    #Random init state
    init = torch.randint(0, len(trajectory), (Nchains,))
    u_init = get_energies(trajectory, init)
    
    chains = [init, ]
    ratios = []
    accs = []
    
    T=300
    beta = 1/(kB*T)
    
    for step in range(steps):
        i = torch.randint(0, len(trajectory), (Nchains,))
        u = get_energies(trajectory, i)

        ratio = torch.exp(-beta*torch.tensor(u.clone().detach() - u_init.clone().detach()).float())
        ratio = torch.min(ratio, torch.ones_like(ratio))
        acc = torch.rand_like(ratio) < ratio

        i[~acc] = init[~acc]
        u[~acc] = u_init[~acc]

        chains.append(i.clone())
        accs.append(acc)
        ratios.append(ratio)

        init = i.clone()
        u_init = u.clone()
   
    return torch.stack(chains), torch.stack(accs), torch.stack(ratios)