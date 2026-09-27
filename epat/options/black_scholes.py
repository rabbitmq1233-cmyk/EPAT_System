"""Black-Scholes-Merton European option pricing."""

from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def d1_d2(
    s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0
) -> tuple[float, float]:
    """Return the Black-Scholes ``d1`` and ``d2``."""
    if s <= 0 or k <= 0:
        raise ValueError("s and k must be positive")
    if t <= 0 or sigma <= 0:
        raise ValueError("t and sigma must be positive for d1/d2")
    vol = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma**2) * t) / vol
    d2 = d1 - vol
    return d1, d2


def bs_price(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes price of a European call or put.

    ``t`` is time to expiry in years, ``r`` the risk-free rate, ``q`` the
    continuous dividend yield. At/after expiry the intrinsic value is returned.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    disc_r = math.exp(-r * t)
    disc_q = math.exp(-q * t)
    if t <= 0 or sigma <= 0:
        if option_type == "call":
            return float(max(s * disc_q - k * disc_r, 0.0))
        return float(max(k * disc_r - s * disc_q, 0.0))

    d1, d2 = d1_d2(s, k, t, r, sigma, q)
    if option_type == "call":
        return float(s * disc_q * _norm_cdf(d1) - k * disc_r * _norm_cdf(d2))
    return float(k * disc_r * _norm_cdf(-d2) - s * disc_q * _norm_cdf(-d1))
