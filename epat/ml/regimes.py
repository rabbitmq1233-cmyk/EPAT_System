"""Gaussian Mixture Model (1-D) via Expectation-Maximisation.

Used for volatility-regime detection: each observation is assigned to a
"calm" or "stressed" Gaussian with its own mean and variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GMMResult:
    """Fitted one-dimensional Gaussian mixture."""

    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    loglik: float
    labels: np.ndarray
    probs: np.ndarray


def _log_gaussian(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    var = max(var, 1e-12)
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mean) ** 2 / var)


def fit_gmm(
    x,
    n_components: int = 2,
    *,
    max_iter: int = 300,
    tol: float = 1e-8,
    seed: int = 0,
) -> GMMResult:
    """Fit a 1-D Gaussian mixture model with the EM algorithm."""
    values = np.asarray(pd.Series(x, dtype="float64").dropna(), dtype="float64")
    if values.size < n_components * 5:
        raise ValueError("not enough observations to fit the mixture")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    rng = np.random.default_rng(seed)
    # Initialise means by quantiles, weights equally, variances globally.
    quantiles = np.linspace(0, 1, n_components + 2)[1:-1]
    means = np.quantile(values, quantiles) if n_components > 1 else np.array([values.mean()])
    variances = np.full(n_components, values.var() + 1e-8)
    weights = np.full(n_components, 1.0 / n_components)

    prev_ll = -np.inf
    loglik = prev_ll
    resp = np.zeros((values.size, n_components))

    for _ in range(max_iter):
        # E-step: responsibilities.
        log_dens = np.column_stack(
            [
                np.log(max(weights[k], 1e-12))
                + _log_gaussian(values, means[k], variances[k])
                for k in range(n_components)
            ]
        )
        max_log = log_dens.max(axis=1, keepdims=True)
        shifted = np.exp(log_dens - max_log)
        summed = shifted.sum(axis=1, keepdims=True)
        resp = shifted / summed
        loglik = float((max_log + np.log(summed)).sum())

        # M-step: update parameters.
        nk = resp.sum(axis=0) + 1e-12
        weights = nk / values.size
        means = (resp * values[:, None]).sum(axis=0) / nk
        variances = (resp * (values[:, None] - means) ** 2).sum(axis=0) / nk
        variances = np.maximum(variances, 1e-12)

        if abs(loglik - prev_ll) < tol:
            break
        prev_ll = loglik

    # Order components by mean (ascending) for stable labelling.
    order = np.argsort(means)
    means, variances, weights = means[order], variances[order], weights[order]
    resp = resp[:, order]
    labels = resp.argmax(axis=1)

    return GMMResult(
        means=means,
        variances=variances,
        weights=weights,
        loglik=loglik,
        labels=labels,
        probs=resp,
    )
