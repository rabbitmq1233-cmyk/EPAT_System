"""Mean-variance portfolio analytics and allocation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epat.config import TRADING_DAYS


def equal_weights(n_assets: int) -> np.ndarray:
    """Weights for an equally weighted portfolio."""
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")
    return np.full(n_assets, 1.0 / n_assets)


def portfolio_return(weights, mean_returns) -> float:
    """Expected portfolio return ``w' mu``."""
    w = np.asarray(weights, dtype="float64")
    mu = np.asarray(mean_returns, dtype="float64")
    return float(w @ mu)


def portfolio_variance(weights, cov) -> float:
    """Portfolio variance ``w' Sigma w``."""
    w = np.asarray(weights, dtype="float64")
    sigma = np.asarray(cov, dtype="float64")
    return float(w @ sigma @ w)


def portfolio_volatility(weights, cov) -> float:
    """Portfolio standard deviation."""
    return float(np.sqrt(max(portfolio_variance(weights, cov), 0.0)))


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of an asset-return panel."""
    return pd.DataFrame(returns).corr()


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def min_variance_weights(cov) -> np.ndarray:
    """Global minimum-variance weights (fully invested, long/short)."""
    sigma = np.asarray(cov, dtype="float64")
    inv = _safe_inverse(sigma)
    ones = np.ones(sigma.shape[0])
    denom = ones @ inv @ ones
    if denom == 0:
        return equal_weights(sigma.shape[0])
    return inv @ ones / denom


def max_sharpe_weights(mean_returns, cov, risk_free: float = 0.0) -> np.ndarray:
    """Tangency-portfolio weights (maximise Sharpe), solved analytically."""
    mu = np.asarray(mean_returns, dtype="float64")
    sigma = np.asarray(cov, dtype="float64")
    inv = _safe_inverse(sigma)
    excess = mu - risk_free
    raw = inv @ excess
    total = raw.sum()
    if abs(total) < 1e-12:
        return equal_weights(mu.shape[0])
    return raw / total


def kelly_allocation(mean_returns, cov) -> np.ndarray:
    """Continuous Kelly weights ``Sigma^-1 mu`` (not normalised, may sum to
    anything; cap/scale as needed for leverage)."""
    mu = np.asarray(mean_returns, dtype="float64")
    sigma = np.asarray(cov, dtype="float64")
    return _safe_inverse(sigma) @ mu


def random_portfolios(
    mean_returns,
    cov,
    *,
    n_samples: int = 5_000,
    risk_free: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample long-only random portfolios (for efficient-frontier plots)."""
    mu = np.asarray(mean_returns, dtype="float64")
    sigma = np.asarray(cov, dtype="float64")
    n_assets = mu.shape[0]
    rng = np.random.default_rng(seed)

    weights = rng.random((n_samples, n_assets))
    weights /= weights.sum(axis=1, keepdims=True)

    rets = weights @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, sigma, weights))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(vols > 0, (rets - risk_free) / vols, 0.0)

    out = pd.DataFrame(weights, columns=[f"w{i}" for i in range(n_assets)])
    out["return"] = rets
    out["volatility"] = vols
    out["sharpe"] = sharpe
    return out


def efficient_frontier(
    mean_returns,
    cov,
    *,
    n_points: int = 50,
    risk_free: float = 0.0,
) -> pd.DataFrame:
    """Long/short efficient frontier.

    For each target return the minimum-variance portfolio is obtained from the
    standard Lagrangian solution ``w = Sigma^-1 (lambda*mu + gamma*1)`` with
    ``lambda = (c*target - a) / delta`` and ``gamma = (b - a*target) / delta``,
    where ``a = 1'S^-1 mu``, ``b = mu'S^-1 mu``, ``c = 1'S^-1 1`` and
    ``delta = b*c - a^2``.
    """
    mu = np.asarray(mean_returns, dtype="float64")
    sigma = np.asarray(cov, dtype="float64")
    inv = _safe_inverse(sigma)
    ones = np.ones(mu.shape[0])

    a = float(ones @ inv @ mu)
    b = float(mu @ inv @ mu)
    c = float(ones @ inv @ ones)
    delta = b * c - a * a

    if abs(delta) < 1e-15:
        w = min_variance_weights(cov)
        return pd.DataFrame(
            {
                "target_return": [portfolio_return(w, mu)],
                "return": [portfolio_return(w, mu)],
                "volatility": [portfolio_volatility(w, sigma)],
            }
        )

    mu_min = portfolio_return(min_variance_weights(cov), mu)
    mu_max = float(mu.max())
    lo, hi = min(mu_min, mu_max), max(mu_min, mu_max)

    rows = []
    for target in np.linspace(lo, hi, n_points):
        lam = (c * target - a) / delta
        gamma = (b - a * target) / delta
        w = inv @ (lam * mu + gamma * ones)
        rows.append(
            {
                "target_return": float(target),
                "return": portfolio_return(w, mu),
                "volatility": portfolio_volatility(w, sigma),
            }
        )
    _ = risk_free
    return pd.DataFrame(rows)
