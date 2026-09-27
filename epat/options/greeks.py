"""Option Greeks (Black-Scholes)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from epat.options.black_scholes import _norm_cdf, _norm_pdf, d1_d2


def delta(
    s: float, k: float, t: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0
) -> float:
    """Sensitivity of option price to the underlying price."""
    if t <= 0 or sigma <= 0:
        intrinsic_itm = (s > k) if option_type.lower() == "call" else (s < k)
        sign = 1.0 if option_type.lower() == "call" else -1.0
        return sign * (1.0 if intrinsic_itm else 0.0)
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    disc_q = math.exp(-q * t)
    if option_type.lower() == "call":
        return float(disc_q * _norm_cdf(d1))
    return float(disc_q * (_norm_cdf(d1) - 1.0))


def gamma(
    s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """Rate of change of delta (same for calls and puts)."""
    if t <= 0 or sigma <= 0 or s <= 0:
        return 0.0
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    return float(math.exp(-q * t) * _norm_pdf(d1) / (s * sigma * math.sqrt(t)))


def vega(
    s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """Sensitivity to volatility, per 1.00 (100%) change in volatility."""
    if t <= 0 or sigma <= 0 or s <= 0:
        return 0.0
    d1, _ = d1_d2(s, k, t, r, sigma, q)
    return float(s * math.exp(-q * t) * _norm_pdf(d1) * math.sqrt(t))


def theta(
    s: float, k: float, t: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0
) -> float:
    """Time decay per year (divide by 365 for a calendar-day theta)."""
    if t <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = d1_d2(s, k, t, r, sigma, q)
    first = -(s * math.exp(-q * t) * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(t))
    if option_type.lower() == "call":
        second = -r * k * math.exp(-r * t) * _norm_cdf(d2)
        third = q * s * math.exp(-q * t) * _norm_cdf(d1)
    else:
        second = r * k * math.exp(-r * t) * _norm_cdf(-d2)
        third = -q * s * math.exp(-q * t) * _norm_cdf(-d1)
    return float(first + second + third)


def rho(
    s: float, k: float, t: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0
) -> float:
    """Sensitivity to the risk-free rate, per 1.00 change in rate."""
    if t <= 0 or sigma <= 0:
        return 0.0
    _, d2 = d1_d2(s, k, t, r, sigma, q)
    if option_type.lower() == "call":
        return float(k * t * math.exp(-r * t) * _norm_cdf(d2))
    return float(-k * t * math.exp(-r * t) * _norm_cdf(-d2))


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def option_greeks(
    s: float, k: float, t: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0
) -> Greeks:
    """All first/second-order Greeks in one call."""
    return Greeks(
        delta=delta(s, k, t, r, sigma, option_type, q),
        gamma=gamma(s, k, t, r, sigma, q),
        vega=vega(s, k, t, r, sigma, q),
        theta=theta(s, k, t, r, sigma, option_type, q),
        rho=rho(s, k, t, r, sigma, option_type, q),
    )
