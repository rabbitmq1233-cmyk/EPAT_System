"""Donchian / Turtle breakout strategy state machine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epat.strategies.base import register


@register("breakout_atr")
def breakout_atr(
    data: pd.DataFrame,
    entry_n: int = 20,
    exit_n: int = 10,
    allow_short: bool = True,
) -> pd.Series:
    """Turtle-style breakout: enter on an ``entry_n``-bar high/low break,
    exit on an ``exit_n``-bar low/high break."""
    if entry_n < 2 or exit_n < 1:
        raise ValueError("entry_n must be >= 2 and exit_n >= 1")

    high, low, close = data["high"], data["low"], data["close"]
    entry_hi = high.rolling(entry_n, min_periods=entry_n).max().shift(1).to_numpy()
    entry_lo = low.rolling(entry_n, min_periods=entry_n).min().shift(1).to_numpy()
    exit_hi = high.rolling(exit_n, min_periods=exit_n).max().shift(1).to_numpy()
    exit_lo = low.rolling(exit_n, min_periods=exit_n).min().shift(1).to_numpy()
    c = close.to_numpy()

    n = len(c)
    state = np.zeros(n)
    pos = 0
    for i in range(n):
        if pos == 0:
            if np.isfinite(entry_hi[i]) and c[i] > entry_hi[i]:
                pos = 1
            elif allow_short and np.isfinite(entry_lo[i]) and c[i] < entry_lo[i]:
                pos = -1
        elif pos == 1:
            if np.isfinite(exit_lo[i]) and c[i] < exit_lo[i]:
                pos = -1 if (allow_short and np.isfinite(entry_lo[i]) and c[i] < entry_lo[i]) else 0
        else:  # pos == -1
            if np.isfinite(exit_hi[i]) and c[i] > exit_hi[i]:
                pos = 1 if (np.isfinite(entry_hi[i]) and c[i] > entry_hi[i]) else 0
        state[i] = pos
    return pd.Series(state, index=close.index, name="signal")
