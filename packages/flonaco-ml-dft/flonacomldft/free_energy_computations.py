import numpy as np
from scipy.special import logsumexp
from scipy.optimize import root_scalar
import warnings


def compute_TFP(n_prop, target_log_prob, prop):
    r"""
    Compute the Importance Sampling estimator of the log partition of the target 
    from normalized prop distribution. This is also known as Targeted Free Energy 
    Perturbation by Jarzinsky which is using a transport map.

    Parameters
    ----------
    target_log_prob : callable
        Log probability of the target distribution.
    prop : Flow or torch.distributions.Distribution
        Proposal distribution, with .sample and .log_prob methods.
    n_prop : int, optional
        Number of samples from the proposal distribution. If None, use the same number as the target.

    Returns
    -------
    logr : ``float``
        Estimate of log partition of target.
    logr_err : ``float``
        Estimate of the variance of the log partition computation.
    """
    xs_prop = prop.sample(n_prop)

    logprob_prop = - prop.nll(xs_prop).detach().numpy()

    logprob_target = target_log_prob(xs_prop).detach().numpy().squeeze() 
    # squeezing because the mlp are returning a 2d tensor with a size 1 extra column

    logr = - logsumexp(logprob_target - logprob_prop) + np.log(n_prop)

    # TODO: rewrite with logsumexp - no log?
    logr_err = np.var(np.exp(logprob_target - logprob_prop)) / (logr ** 2) / n_prop

    return logr, np.sqrt(logr_err)


def compute_TFP_logratio(n_prop, target_log_prob, mix_prop):
    """Compute the log ratio of the TFP estimator.
    Output: 
        - logr: beta * F_0 - beta * F_1 = log(weight_0 / weight(1))
    """
    logrs = []
    logr_errs = []

    for mode in range(2):

        prop = mix_prop[mode]

        if type(target_log_prob) == list:
            target_log_p = target_log_prob[mode]
        else:
            target_log_p = target_log_prob

        logr, logr_err = compute_TFP(n_prop, target_log_p, prop)

        logrs.append(logr)
        logr_errs.append(logr_err)
        
    return  logrs[0] - logrs[1], logr_errs[0] + logr_errs[1], logrs


def compute_BAR(xs, target_log_prob, prop, n_prop=None, 
        maxiter=2000, rtol=1e-13, xtol=1e-13, thin=1, ess=1
        ):
    r"""
    Bridge Sampling/Bennett Acceptance Ratio estimator for the log partition of target.
    Refs: https://arxiv.org/abs/1912.06073, pocomc, deepbar
    Parameters
    ----------
    xs : torch.Tensor
        Samples from the target distribution.
    target_log_prob : callable
        Log probability of the target distribution.
    prop : Flow or torch.distributions.Distribution
        Proposal distribution, with .sample and .log_prob methods.
    n_prop : int, optional
        Number of samples from the proposal distribution. If None, use the same number as the target.
    maxiter : ``int``
        Maximum number of iterations of root-finding procedure.
    rtol : ``float``
        Relative numerical tolerance of root-finding procedure.
    xtol : ``float``
        Absolute numerical tolerance of root-finding procedure.
    thin : ``int``
        Thin the samples by a integer factor. Default is ``thin=1``
        (no thinning).
    ess : ``float``
        Effective sample size of xs samples to estimate errors.
    
    Returns
    -------
    logr : ``float``
        Estimate of log partition of target.
    logr_err : ``float``
        Estimate of the variance of the log partition computation.
    """

    if n_prop is None:
        n_prop = len(xs)
    xs_prop = prop.sample(n_prop).detach()

    xs = xs[:thin]

    n_mcmc = len(xs)
    n_prop = len(xs_prop)

    logprop_prop = - prop.nll(xs_prop).detach().numpy() 
    logprop_mc = - prop.nll(xs).detach().numpy() 
    logtgt_prop = target_log_prob(xs_prop).detach().numpy()
    logtgt_mc = target_log_prob(xs).detach().numpy()
    if len(logtgt_prop.shape) > 1:
        logtgt_prop = logtgt_prop[:, 0] 
        logtgt_mc = logtgt_mc[:, 0]

    _a = logprop_mc - logtgt_mc - np.log(n_mcmc / n_prop)
    _b = logtgt_prop - logprop_prop + np.log(n_mcmc / n_prop)

    def score(logr):
        _c = logsumexp(logr + _a - logsumexp(np.array((logr + _a,
                    np.zeros_like(_a))), axis=0))
        _d = logsumexp(-logr + _b - logsumexp(np.array((-logr + _b,
                    np.zeros_like(_b))), axis=0))
        return _c - _d

    logr = root_scalar(score, x0=0, x1=5, maxiter=maxiter, rtol=rtol, xtol=xtol).root

    # Uncertainty
    s1 = n_mcmc / (n_mcmc + n_prop)
    s2 = n_prop / (n_mcmc + n_prop)

    f1 = np.exp(logtgt_prop - logr - logsumexp(np.array((logtgt_prop - logr +
                np.log(s1), logprop_prop + np.log(s2))), axis=0))
    f2 = np.exp(logprop_mc - logsumexp(np.array((logtgt_mc - logr +
                np.log(s1), logprop_mc + np.log(s2))), axis=0))
    re2_q = np.var(f1) / np.mean(f1)**2 / n_prop

    tau = 1.0 / ess
    re2_p = tau * np.var(f2) / np.mean(f2)**2 / n_mcmc
    logr_err = (re2_p + re2_q)**0.5

    return - logr, logr_err


def compute_deepBAR_logratio(xs, cs, target_log_prob, mix_prop, n_prop=None):
    """Compute the log ratio of the deepBAR estimator.
    Output: 
        - logr: beta * F_0 - beta * F_1 = log(weight_0 / weight(1))
    """
    logrs = []
    logr_errs = []

    for mode in range(2):

        prop = mix_prop[mode]

        xs_mode = xs[cs == mode]
        
        # TODO: plugin true ESS to compute the error
        warnings.warn("Error estimation misses ESS estimation")
        ess = 1

        if type(target_log_prob) == list:
            target_log_p = target_log_prob[mode]
        else:
            target_log_p = target_log_prob

        logr, logr_err = compute_BAR(xs_mode, target_log_p, prop, n_prop=n_prop,
                                     ess=ess)

        logrs.append(logr)
        logr_errs.append(logr_err)
        
    return  logrs[0] - logrs[1], logr_errs[0] + logr_errs[1], logrs 
