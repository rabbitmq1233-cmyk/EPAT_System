"""Option strategy payoff diagrams (at expiry)."""

from __future__ import annotations

import numpy as np


def straddle_payoff(
    s_range: np.ndarray, k: float, premium: float, long: bool = True
) -> np.ndarray:
    """Payoff of a long/short straddle (call + put at strike ``k``)."""
    s = np.asarray(s_range, dtype="float64")
    edge = np.abs(s - k)
    value = edge - premium
    return value if long else -value


def strangle_payoff(
    s_range: np.ndarray,
    k_put: float,
    k_call: float,
    premium: float,
    long: bool = True,
) -> np.ndarray:
    """Payoff of a long/short strangle (OTM put + OTM call)."""
    if k_put >= k_call:
        raise ValueError("k_put must be below k_call for a strangle")
    s = np.asarray(s_range, dtype="float64")
    edge = np.maximum(k_put - s, 0.0) + np.maximum(s - k_call, 0.0)
    value = edge - premium
    return value if long else -value


def butterfly_payoff(
    s_range: np.ndarray,
    k_low: float,
    k_mid: float,
    k_high: float,
    net_debit: float,
) -> np.ndarray:
    """Payoff of a long call butterfly (buy low/high, sell two middles)."""
    if not k_low < k_mid < k_high:
        raise ValueError("strikes must satisfy k_low < k_mid < k_high")
    s = np.asarray(s_range, dtype="float64")
    value = (
        np.maximum(s - k_low, 0.0)
        - 2.0 * np.maximum(s - k_mid, 0.0)
        + np.maximum(s - k_high, 0.0)
    )
    return value - net_debit


def straddle_breakevens(k: float, premium: float) -> tuple[float, float]:
    """Lower and upper breakevens of a long straddle."""
    return float(k - premium), float(k + premium)
