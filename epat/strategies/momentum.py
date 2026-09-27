"""Time-series momentum and futures roll-return strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epat.strategies.base import close_series, register


@register("momentum")
def momentum(
    data,
    lookback: int = 60,
    allow_short: bool = True,
    long_only: bool = False,
    volatility_scaled: bool = False,
) -> pd.Series:
    """Time-series momentum: go long if the past ``lookback`` return is positive.

    When ``volatility_scaled`` is set the position is scaled inversely to
    trailing volatility (capped at 1x).
    """
    close = close_series(data)
    past = close.pct_change(lookback)
    signal = np.sign(past)
    if volatility_scaled:
        vol = close.pct_change().rolling(lookback, min_periods=lookback).std(ddof=1)
        scale = (0.15 / (vol * np.sqrt(252))).clip(upper=1.0)
        signal = signal * scale
    if long_only:
        signal = signal.clip(lower=0.0)
    elif not allow_short:
        signal = signal.clip(lower=0.0)
    return signal.fillna(0.0).rename("signal")


def roll_return(futures: pd.Series, spot: pd.Series) -> pd.Series:
    """Futures roll return = total futures return minus spot return (log)."""
    f = pd.Series(futures, dtype="float64")
    s = pd.Series(spot, dtype="float64")
    frame = pd.concat([f.rename("f"), s.rename("s")], axis=1, sort=False).dropna()
    total = np.log(frame["f"]).diff()
    spot_ret = np.log(frame["s"]).diff()
    return (total - spot_ret).rename("roll_return")


@register("roll_return_momentum")
def roll_return_momentum(
    futures: pd.Series,
    spot: pd.Series,
    lookback: int = 60,
    allow_short: bool = True,
) -> pd.Series:
    """Momentum on the cumulative roll return of a futures contract."""
    roll = roll_return(futures, spot)
    cumulative = roll.rolling(lookback, min_periods=max(2, lookback // 4)).sum()
    signal = np.sign(cumulative)
    if not allow_short:
        signal = signal.clip(lower=0.0)
    return signal.fillna(0.0).rename("signal")
