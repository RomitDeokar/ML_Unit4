"""From-scratch scaled sum-product and log-domain Viterbi for Gaussian HMMs.

The emission row shift and forward scaling prevent underflow on long series.
These functions do not call hmmlearn's inference routines.
"""
import numpy as np


def emissions(x, means, variances):
    variances = np.maximum(variances, 1e-6)
    return -.5 * (np.log(2 * np.pi * variances).sum(axis=1)[None, :]
                  + (((x[:, None, :] - means[None, :, :]) ** 2)
                     / variances[None, :, :]).sum(axis=2))


def forward_backward(log_emission, start, transition):
    """Return causal alpha, retrospective gamma, log likelihood, scale factors."""
    if len(log_emission) == 0:
        raise ValueError('An observation sequence must not be empty.')
    shift = log_emission.max(axis=1)
    b = np.exp(log_emission - shift[:, None])
    n, k = b.shape
    alpha = np.zeros((n, k))
    scales = np.zeros(n)
    alpha[0] = start * b[0]
    scales[0] = max(alpha[0].sum(), np.finfo(float).tiny)
    alpha[0] /= scales[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ transition) * b[t]
        scales[t] = max(alpha[t].sum(), np.finfo(float).tiny)
        alpha[t] /= scales[t]
    beta = np.ones((n, k))
    for t in range(n - 2, -1, -1):
        beta[t] = transition @ (b[t + 1] * beta[t + 1]) / scales[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return alpha, gamma, float(np.sum(np.log(scales) + shift)), scales


def viterbi(log_emission, start, transition):
    """Maximum a posteriori state path, with backpointers."""
    log_a = np.log(np.maximum(transition, 1e-300))
    delta = log_emission[0] + np.log(np.maximum(start, 1e-300))
    back = np.zeros(log_emission.shape, dtype=int)
    for t in range(1, len(log_emission)):
        candidates = delta[:, None] + log_a
        back[t] = candidates.argmax(axis=0)
        delta = candidates.max(axis=0) + log_emission[t]
    path = np.zeros(len(log_emission), dtype=int)
    path[-1] = delta.argmax()
    for t in range(len(path) - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def horizon_risk(alpha, transition, critical, hours=6):
    """Probability of occupying/reaching critical by horizon; not calibrated."""
    absorbing = transition.copy()
    absorbing[critical] = 0
    absorbing[critical, critical] = 1
    return np.clip(alpha @ np.linalg.matrix_power(absorbing, hours)[:, critical], 0, 1)
