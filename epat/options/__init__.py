"""Options analytics: pricing, Greeks, implied vol, vol estimators, payoffs."""

from __future__ import annotations

from epat.options.black_scholes import bs_price, d1_d2
from epat.options.greeks import (
    delta,
    gamma,
    option_greeks,
    rho,
    theta,
    vega,
)
from epat.options.implied_vol import implied_vol
from epat.options.parity import (
    parity_gap,
    synthetic_call,
    synthetic_put,
)
from epat.options.payoffs import (
    butterfly_payoff,
    straddle_breakevens,
    strangle_payoff,
    straddle_payoff,
)
from epat.options.variance_premium import (
    premium_stats,
    realised_vol,
    variance_premium,
)
from epat.options.vol_estimators import (
    GARCHResult,
    ewma_vol,
    garch11_fit,
)

__all__ = [
    "bs_price",
    "d1_d2",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "option_greeks",
    "implied_vol",
    "parity_gap",
    "synthetic_call",
    "synthetic_put",
    "straddle_payoff",
    "strangle_payoff",
    "butterfly_payoff",
    "straddle_breakevens",
    "realised_vol",
    "variance_premium",
    "premium_stats",
    "ewma_vol",
    "garch11_fit",
    "GARCHResult",
]
