"""Cointegration-based pairs trading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from epat.indicators import cointegration, zscore
from epat.indicators.stats import CointegrationResult
from epat.strategies.base import register
from epat.strategies.mean_reversion import _reversion_state


@dataclass
class PairsResult:
    """Output of the pairs strategy."""

    signal: pd.Series
    spread: pd.Series
    spread_returns: pd.Series
    hedge_ratio: float
    coint: CointegrationResult


@register("pairs")
def pairs(
    y: pd.Series,
    x: pd.Series,
    *,
    train: int = 120,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    allow_short: bool = True,
) -> PairsResult:
    """Engle-Granger pairs trade on ``spread = y - beta * x``.

    ``beta`` (hedge ratio) is estimated on the first ``train`` observations to
    avoid look-ahead, then the spread z-score drives entries/exits.
    """
    yv = pd.Series(y, dtype="float64")
    xv = pd.Series(x, dtype="float64")
    frame = pd.concat([yv.rename("y"), xv.rename("x")], axis=1, sort=False).dropna()
    if frame.shape[0] < max(train, window) + 10:
        raise ValueError("not enough overlapping observations for pairs trading")

    fit_n = max(20, min(train, frame.shape[0] // 2))
    coint_fit = cointegration(frame["y"].iloc[:fit_n], frame["x"].iloc[:fit_n])
    beta = coint_fit.hedge_ratio

    spread = (frame["y"] - beta * frame["x"]).rename("spread")
    z = zscore(spread, window)
    signal = _reversion_state(z, entry_z, exit_z, allow_short)

    ret_y = frame["y"].pct_change().fillna(0.0)
    ret_x = frame["x"].pct_change().fillna(0.0)
    spread_returns = (ret_y - beta * ret_x).rename("spread_returns")

    return PairsResult(
        signal=signal,
        spread=spread,
        spread_returns=spread_returns,
        hedge_ratio=beta,
        coint=coint_fit,
    )
