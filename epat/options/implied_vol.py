"""Implied-volatility solver (bisection)."""

from __future__ import annotations

from epat.options.black_scholes import bs_price


def implied_vol(
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
    lo: float = 1e-6,
    hi: float = 5.0,
) -> float:
    """Invert the Black-Scholes price for volatility via bisection.

    Returns ``nan`` when the price violates no-arbitrage bounds or the root
    cannot be bracketed within ``[lo, hi]``.
    """
    if t <= 0:
        return float("nan")
    intrinsic = bs_price(s, k, t, r, 0.0, option_type, q)
    if price < intrinsic - 1e-8:
        return float("nan")

    f_lo = bs_price(s, k, t, r, lo, option_type, q) - price
    f_hi = bs_price(s, k, t, r, hi, option_type, q) - price
    if f_lo * f_hi > 0:
        return float("nan")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(s, k, t, r, mid, option_type, q) - price
        if abs(f_mid) < tol or (hi - lo) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))
