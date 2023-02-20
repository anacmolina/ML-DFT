import numpy as np
import torch

def get_participation_ratio(prop, target_log_prob, n_prop):
    """
    Compute the participation ratio of a proposal distribution

    Args:
        prop : Proposal distribution with .sample and .nll methods
        target_log_prob : Log probability of the target distribution.
        n_prop : Number of samples from the proposal distribution.
    
    Returns:
        Participation ratio, 1 if proposal is equal to target and < 1 otherwise

    """
    xs = prop.sample(n_prop)
    log_weight = target_log_prob(xs).squeeze() + prop.nll(xs)
    log_ratio = torch.logsumexp(2 * log_weight, dim=0) - 2 * torch.logsumexp(log_weight, dim=0) 
    return torch.exp(-log_ratio) / n_prop

def get_ESS(x):
    """
    Patching to take convention of axis orders,
    and convert from torch to numpy
    x : (n_iter, m_chaines, dim)
    """
    try:
        x = x.detach().numpy()
    except AttributeError:
        x = x

    x = x.swapaxes(0, 1)
    return my_ESS(x)


def my_ESS(x):
    """
    Compute the effective sample size of estimand of interest.
    Vectorised implementation.
    x : m_chains, n_iter, dim
    Computation of effective sampling size from:
    https://github.com/jwalton3141/jwalton3141.github.io
    following definition from:
    ref Gelman, Andrew, J. B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, 
    and Donald B. Rubin. 2013. Bayesian Data Analysis. Third Edition. London: Chapman & Hall / CRC Press.
    """
    if x.shape < (2,):
        raise ValueError(
            'Calculation of effective sample size'
            'requires multiple chains of the same length.')
    try:
        m_chains, n_iter = x.shape
    except ValueError:
        return [my_ESS(y.T) for y in x.T]

    def variogram(t): return (
        (x[:, t:] - x[:, :(n_iter - t)])**2).sum() / (m_chains * (n_iter - t))

    post_var = my_gelman_rubin(x)
    assert post_var > 0

    t = 1
    rho = np.ones(n_iter)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iter):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t - 1:t + 1]) < 0

        t += 1

    return int(m_chains * n_iter / (1 + 2 * rho[1:t].sum()))


def my_gelman_rubin(x):
    """
    Estimate the marginal posterior variance. Vectorised implementation.
    x : m_chaines, n_iter
    """
    m_chains, n_iter = x.shape

    # Calculate between-chain variance
    B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True)) **
         2).sum() / (m_chains * (n_iter - 1))

    # (over) estimate of variance
    s2 = W * (n_iter - 1) / n_iter + B_over_n

    return s2