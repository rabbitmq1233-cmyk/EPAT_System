"""Mean-reversion strategies (z-score and Bollinger)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epat.indicators import zscore
from epat.strategies.base import close_series, register


def _reversion_state(z: pd.Series, entry_z: float, exit_z: float, allow_short: bool) -> pd.Series:
    zv = z.to_numpy()
    n = zv.size
    state = np.zeros(n)
    pos = 0
    for i in range(n):
        zi = zv[i]
        if not np.isfinite(zi):
            state[i] = pos
            continue
        if pos == 0:
            if zi <= -entry_z:
                pos = 1
            elif allow_short and zi >= entry_z:
                pos = -1
        elif pos == 1:
            if zi >= -exit_z:
                pos = 0
        else:  # short
            if zi <= exit_z:
                pos = 0
        state[i] = pos
    return pd.Series(state, index=z.index, name="signal")


@register("mean_reversion")
def mean_reversion(
    data,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    allow_short: bool = True,
) -> pd.Series:
    """Buy when price z-score < -entry_z, sell when > +entry_z, exit near the mean."""
    if entry_z <= exit_z:
        raise ValueError("entry_z must be greater than exit_z")
    close = close_series(data)
    z = zscore(close, window)
    return _reversion_state(z, entry_z, exit_z, allow_short)


@register("bollinger_reversion")
def bollinger_reversion(
    data,
    window: int = 20,
    k: float = 2.0,
    exit_z: float = 0.0,
    allow_short: bool = True,
) -> pd.Series:
    """Bollinger mean reversion: fade moves beyond ``k`` standard deviations."""
    return mean_reversion(
        data, window=window, entry_z=k, exit_z=exit_z, allow_short=allow_short
    )
