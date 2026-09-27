"""Moving-average crossover strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epat.indicators import ema, sma
from epat.strategies.base import close_series, register


@register("ma_crossover")
def ma_crossover(
    data,
    fast: int = 10,
    slow: int = 30,
    long_only: bool = False,
    kind: str = "sma",
) -> pd.Series:
    """Long when the fast MA is above the slow MA, short otherwise.

    ``kind`` selects ``"sma"`` or ``"ema"``. With ``long_only=True`` the
    strategy holds flat instead of shorting.
    """
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")
    close = close_series(data)
    mover = sma if kind == "sma" else ema
    fast_ma = mover(close, fast)
    slow_ma = mover(close, slow)

    raw = np.where(fast_ma.to_numpy() > slow_ma.to_numpy(), 1.0, -1.0)
    signal = pd.Series(raw, index=close.index, name="signal")
    signal[fast_ma.isna() | slow_ma.isna()] = 0.0
    if long_only:
        signal = signal.clip(lower=0.0)
    return signal
