"""Volatility / range indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epat.config import TRADING_DAYS


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range = max(H-L, |H-prevClose|, |prevClose-L|)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    components = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
        sort=False,
    )
    return components.max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range using the recursive/EMA form used by Turtle traders.

    ``ATR_t = (ATR_{t-1} * (n-1) + TR_t) / n`` which equals an EWM with
    ``alpha = 1/n``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rolling_std(series: pd.Series, n: int) -> pd.Series:
    """Rolling population standard deviation."""
    if n < 2:
        raise ValueError("n must be >= 2")
    series = pd.Series(series, dtype="float64")
    return series.rolling(window=n, min_periods=n).std(ddof=0)


def bollinger(
    close: pd.Series, n: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(middle, upper, lower)`` Bollinger bands."""
    close = pd.Series(close, dtype="float64")
    mid = close.rolling(window=n, min_periods=n).mean()
    sd = close.rolling(window=n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def close_to_close_vol(
    close: pd.Series, n: int = 30, ann: int = TRADING_DAYS
) -> pd.Series:
    """Annualised close-to-close volatility of log returns."""
    close = pd.Series(close, dtype="float64")
    return np.log(close).diff().rolling(window=n, min_periods=n).std(ddof=1) * np.sqrt(ann)


def parkinson_vol(
    df: pd.DataFrame, n: int = 30, ann: int = TRADING_DAYS
) -> pd.Series:
    """Annualised Parkinson high-low volatility estimator.

    ``sigma^2 = mean(ln(H/L)^2) / (4 ln 2)`` annualised by ``ann``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    hl2 = np.log(df["high"] / df["low"]) ** 2
    var = hl2.rolling(window=n, min_periods=n).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * ann)
