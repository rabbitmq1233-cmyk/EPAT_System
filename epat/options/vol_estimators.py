"""Volatility estimation and forecasting: EWMA and GARCH(1,1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from epat.config import TRADING_DAYS


def ewma_vol(
    returns: pd.Series, lam: float = 0.94, ann: int = TRADING_DAYS
) -> pd.Series:
    """RiskMetrics-style exponentially weighted volatility (annualised)."""
    ret = pd.Series(returns, dtype="float64").dropna()
    var = ret.pow(2).ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(var * ann).rename("ewma_vol")


@dataclass
class GARCHResult:
    """Fitted GARCH(1,1) parameters and conditional volatility."""

    omega: float
    alpha: float
    beta: float
    persistence: float
    unconditional_vol: float
    conditional_vol: pd.Series
    forecast_vol: float
    loglik: float
    annualisation: int

    def forecast(self, horizon: int = 1) -> float:
        """Annualised multi-step volatility forecast."""
        var_t = (self.conditional_vol.iloc[-1] / np.sqrt(self.annualisation)) ** 2
        long_run = (self.omega / max(1e-12, 1.0 - self.persistence))
        var_h = long_run + (self.persistence ** horizon) * (var_t - long_run)
        return float(np.sqrt(var_h * self.annualisation))


def _conditional_var(r: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    n = r.size
    var = np.empty(n, dtype="float64")
    var[0] = r.var() if r.var() > 0 else 1e-8
    for t in range(1, n):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
    return np.maximum(var, 1e-12)


def _loglik(r: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return float("-inf")
    var = _conditional_var(r, omega, alpha, beta)
    return float(-0.5 * np.sum(np.log(2.0 * np.pi * var) + r**2 / var))


def garch11_fit(
    returns: pd.Series,
    ann: int = TRADING_DAYS,
    *,
    refine: bool = True,
) -> GARCHResult:
    """Fit a GARCH(1,1) model by (coarse) maximum likelihood.

    Uses a two-stage grid search so no SciPy dependency is required. Good
    enough for volatility-filtering and regime work; use the ``arch`` package
    for production-grade estimation.
    """
    ret = pd.Series(returns, dtype="float64").dropna()
    if ret.size < 50:
        raise ValueError("need at least 50 return observations to fit GARCH")
    r = ret.to_numpy()
    sample_var = float(np.var(r))

    best = (float("-inf"), 0.05, 0.90)
    for alpha in np.linspace(0.02, 0.30, 15):
        for beta in np.linspace(0.40, 0.97, 20):
            if alpha + beta >= 0.999:
                continue
            omega = sample_var * (1.0 - alpha - beta)
            ll = _loglik(r, omega, alpha, beta)
            if ll > best[0]:
                best = (ll, float(alpha), float(beta))

    if refine:
        ll0, a0, b0 = best
        for alpha in np.linspace(max(0.01, a0 - 0.05), min(0.4, a0 + 0.05), 11):
            for beta in np.linspace(max(0.3, b0 - 0.05), min(0.985, b0 + 0.05), 11):
                if alpha + beta >= 0.999:
                    continue
                omega = sample_var * (1.0 - alpha - beta)
                ll = _loglik(r, omega, alpha, beta)
                if ll > best[0]:
                    best = (ll, float(alpha), float(beta))

    loglik, alpha, beta = best
    omega = sample_var * (1.0 - alpha - beta)
    cond_var = _conditional_var(r, omega, alpha, beta)
    cond_vol = pd.Series(np.sqrt(cond_var * ann), index=ret.index, name="garch_vol")
    persistence = alpha + beta
    unconditional = (
        np.sqrt(omega / (1.0 - persistence) * ann) if persistence < 1 else float("inf")
    )
    return GARCHResult(
        omega=float(omega),
        alpha=alpha,
        beta=beta,
        persistence=persistence,
        unconditional_vol=float(unconditional),
        conditional_vol=cond_vol,
        forecast_vol=float(np.sqrt(cond_var[-1] * ann)),
        loglik=float(loglik),
        annualisation=ann,
    )
