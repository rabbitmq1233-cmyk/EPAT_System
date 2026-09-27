"""Synthetic OHLCV generation.

Lets the whole framework run offline for demos and tests, and provides
regime-switching price series for the ML / mixture-model work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epat.config import TRADING_DAYS


def _regime_vol_path(
    n: int,
    rng: np.random.Generator,
    low_vol: float,
    high_vol: float,
    p_stay: float,
) -> np.ndarray:
    """Two-state volatility path (0 = calm, 1 = stressed) with persistence."""
    state = 0
    vols = np.empty(n, dtype="float64")
    for i in range(n):
        vols[i] = low_vol if state == 0 else high_vol
        if rng.random() > p_stay:
            state = 1 - state
    return vols


def generate_ohlcv(
    n: int = 1_000,
    *,
    start: str = "2015-01-01",
    freq: str = "B",
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.20,
    seed: int = 42,
    regime: bool = False,
    high_vol_mult: float = 2.5,
    p_stay: float = 0.97,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV series via geometric Brownian motion.

    Parameters
    ----------
    n:
        Number of bars.
    start, freq:
        Date range start and pandas frequency (``"B"`` = business days).
    s0, mu, sigma:
        Initial price, annual drift and annual volatility.
    regime:
        When ``True``, volatility switches between ``sigma`` and
        ``sigma * high_vol_mult`` with persistence ``p_stay``.
    """
    if n <= 1:
        raise ValueError("n must be greater than 1")

    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    drift = (mu - 0.5 * sigma**2) * dt

    if regime:
        vol = _regime_vol_path(n, rng, sigma, sigma * high_vol_mult, p_stay)
    else:
        vol = np.full(n, sigma, dtype="float64")

    shocks = rng.standard_normal(n) * vol * np.sqrt(dt)
    log_path = np.cumsum(drift + shocks)
    close = s0 * np.exp(log_path)

    prev_close = np.empty(n, dtype="float64")
    prev_close[0] = s0
    prev_close[1:] = close[:-1]
    # Open gaps slightly from the previous close.
    open_ = prev_close * (1.0 + rng.standard_normal(n) * vol * np.sqrt(dt) * 0.3)

    intraday = np.abs(rng.standard_normal(n)) * vol * np.sqrt(dt)
    high = np.maximum(open_, close) * (1.0 + intraday * 0.5)
    low = np.minimum(open_, close) * (1.0 - intraday * 0.5)
    low = np.minimum(low, np.minimum(open_, close))
    high = np.maximum(high, np.maximum(open_, close))

    base_volume = 1_000_000
    volume = base_volume * np.exp(rng.standard_normal(n) * 0.4)

    index = pd.date_range(start=start, periods=n, freq=freq, name="date")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def generate_multi(
    n_assets: int = 3,
    n: int = 1_000,
    *,
    seed: int = 7,
    start: str = "2015-01-01",
    avg_corr: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Generate correlated synthetic assets (for portfolio / PCA work).

    Returns a mapping ``symbol -> OHLCV DataFrame``.
    """
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")

    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS

    # Correlated daily log-return shocks.
    corr = np.full((n_assets, n_assets), avg_corr)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    mus = np.linspace(0.05, 0.12, n_assets)
    sigmas = np.linspace(0.15, 0.30, n_assets)

    shocks = rng.standard_normal((n, n_assets)) @ chol.T
    out: dict[str, pd.DataFrame] = {}
    for j in range(n_assets):
        drift = (mus[j] - 0.5 * sigmas[j] ** 2) * dt
        log_path = np.cumsum(drift + shocks[:, j] * sigmas[j] * np.sqrt(dt))
        close = 100.0 * np.exp(log_path)
        prev = np.concatenate([[100.0], close[:-1]])
        intraday = np.abs(rng.standard_normal(n)) * sigmas[j] * np.sqrt(dt)
        high = np.maximum(prev, close) * (1.0 + intraday * 0.5)
        low = np.minimum(prev, close) * (1.0 - intraday * 0.5)
        index = pd.date_range(start=start, periods=n, freq="B", name="date")
        out[f"SYN{j + 1}"] = pd.DataFrame(
            {
                "open": prev,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1_000_000 * np.exp(rng.standard_normal(n) * 0.4),
            },
            index=index,
        )
    return out
