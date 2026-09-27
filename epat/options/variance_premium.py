"""Variance-premium analytics (implied vs realised volatility)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epat.config import TRADING_DAYS


def realised_vol(
    returns: pd.Series, window: int = 21, ann: int = TRADING_DAYS
) -> pd.Series:
    """Rolling annualised realised volatility of returns."""
    ret = pd.Series(returns, dtype="float64")
    return ret.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(ann)


def variance_premium(
    implied_vol: pd.Series, realised_volatility: pd.Series
) -> pd.Series:
    """Volatility premium = implied vol minus subsequent realised vol."""
    iv = pd.Series(implied_vol, dtype="float64")
    rv = pd.Series(realised_volatility, dtype="float64")
    aligned = pd.concat([iv.rename("iv"), rv.rename("rv")], axis=1, sort=False).dropna()
    return (aligned["iv"] - aligned["rv"]).rename("variance_premium")


def premium_stats(
    implied_vol: pd.Series, realised_volatility: pd.Series
) -> dict[str, float]:
    """Summary statistics of the variance premium.

    The mean premium as a fraction of the implied level is the practitioner
    rule of thumb (often ~20% for equity indices).
    """
    prem = variance_premium(implied_vol, realised_volatility)
    if prem.empty:
        return {
            "mean_premium": float("nan"),
            "median_premium": float("nan"),
            "premium_pct_of_iv": float("nan"),
            "share_positive": float("nan"),
            "n_obs": 0.0,
        }
    iv = pd.Series(implied_vol, dtype="float64").reindex(prem.index)
    mean_premium = float(prem.mean())
    mean_iv = float(iv.mean())
    return {
        "mean_premium": mean_premium,
        "median_premium": float(prem.median()),
        "premium_pct_of_iv": float(mean_premium / mean_iv) if mean_iv else float("nan"),
        "share_positive": float((prem > 0).mean()),
        "n_obs": float(prem.size),
    }
