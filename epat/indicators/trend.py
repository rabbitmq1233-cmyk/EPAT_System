"""Trend indicators: simple and exponential moving averages."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average over ``n`` periods."""
    if n < 1:
        raise ValueError("n must be >= 1")
    series = pd.Series(series, dtype="float64")
    return series.rolling(window=n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential moving average with span ``n`` (recursive form)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    series = pd.Series(series, dtype="float64")
    return series.ewm(span=n, adjust=False, min_periods=n).mean()
