"""Position-sizing methods.

Covers the Kelly family (discrete and continuous), fixed-fractional sizing,
volatility targeting and ATR-based (fixed-risk) sizing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epat.config import TRADING_DAYS


def kelly_binary(win_prob: float, payoff_ratio: float, loss_fraction: float = 1.0) -> float:
    """Kelly fraction for a binary bet.

    ``f = (p*b - a*q) / (a*b)`` where ``p`` is the win probability, ``q = 1-p``,
    ``b`` the win payoff per unit risked and ``a`` the loss per unit risked.
    With even odds (``b = a = 1``) this reduces to ``f = 2p - 1``.
    """
    if not 0.0 <= win_prob <= 1.0:
        raise ValueError("win_prob must be in [0, 1]")
    if payoff_ratio <= 0 or loss_fraction <= 0:
        raise ValueError("payoff_ratio and loss_fraction must be positive")
    q = 1.0 - win_prob
    f = (win_prob * payoff_ratio - loss_fraction * q) / (loss_fraction * payoff_ratio)
    return float(max(0.0, f))


def kelly_continuous(expected_return: float, variance: float) -> float:
    """Continuous Kelly fraction ``f* = mu / sigma^2`` (fractional of wealth)."""
    if variance <= 0:
        raise ValueError("variance must be positive")
    return float(max(0.0, expected_return / variance))


def fractional_kelly(full_kelly: float, fraction: float = 0.5) -> float:
    """Scale a full-Kelly fraction down (e.g. half-Kelly to damp drawdowns)."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return float(max(0.0, full_kelly) * fraction)


def fixed_fraction_size(equity: float, price: float, fraction: float) -> float:
    """Number of units to buy using a fixed fraction of equity."""
    if price <= 0:
        raise ValueError("price must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return float(equity * fraction / price)


def volatility_target(
    returns: pd.Series,
    target_vol: float = 0.15,
    window: int = 20,
    periods_per_year: int = TRADING_DAYS,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Leverage series that targets an annualised volatility.

    Returns ``target_vol / realised_vol`` clipped to ``[0, max_leverage]``.
    """
    ret = pd.Series(returns, dtype="float64")
    realised = ret.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)
    leverage = (target_vol / realised.replace(0.0, np.nan)).clip(upper=max_leverage)
    return leverage.fillna(0.0)


def atr_position_size(
    equity: float,
    atr_value: float,
    risk_per_trade: float = 0.01,
    atr_multiplier: float = 2.0,
) -> float:
    """Units sized so a ``atr_multiplier * ATR`` move risks ``risk_per_trade`` of equity."""
    if atr_value <= 0:
        return 0.0
    risk_amount = equity * risk_per_trade
    stop_distance = atr_multiplier * atr_value
    if stop_distance <= 0:
        return 0.0
    return float(max(0.0, risk_amount / stop_distance))
