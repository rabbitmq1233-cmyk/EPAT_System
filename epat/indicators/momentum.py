"""Momentum indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing).

    Values above 70 are conventionally "overbought" and below 30 "oversold".
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    close = pd.Series(close, dtype="float64")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))

    # Edge cases: no losses -> 100, no movement at all -> neutral 50.
    flat = (avg_gain == 0) & (avg_loss == 0)
    only_gains = (avg_loss == 0) & (avg_gain > 0)
    out = out.mask(only_gains, 100.0)
    out = out.mask(flat, 50.0)
    return out.replace([np.inf, -np.inf], np.nan)
