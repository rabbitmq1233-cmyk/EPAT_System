"""Put-call parity and synthetic positions."""

from __future__ import annotations

import math


def parity_gap(call: float, put: float, s: float, k: float, t: float, r: float) -> float:
    """Difference between the call and the parity-implied call.

    ``C - P = S - K e^{-rT}``. A positive gap suggests the call is rich
    (or the put cheap) relative to parity.
    """
    lhs = call - put
    rhs = s - k * math.exp(-r * t)
    return float(lhs - rhs)


def synthetic_call(put: float, s: float, k: float, t: float, r: float) -> float:
    """Replicate a call from a put, the stock and a zero-coupon bond."""
    return float(put + s - k * math.exp(-r * t))


def synthetic_put(call: float, s: float, k: float, t: float, r: float) -> float:
    """Replicate a put from a call, the stock and a zero-coupon bond."""
    return float(call - s + k * math.exp(-r * t))
