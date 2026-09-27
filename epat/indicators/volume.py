"""Volume-weighted indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vwap(df: pd.DataFrame, n: int | None = None) -> pd.Series:
    """Volume-Weighted Average Price.

    ``VWAP = sum(typical_price * volume) / sum(volume)`` using the typical
    price ``(H + L + C) / 3``. When ``n`` is ``None`` the cumulative
    (session-to-date) VWAP is returned; otherwise a rolling ``n``-bar VWAP.
    """
    if "volume" not in df.columns or df["volume"].isna().all():
        raise ValueError("vwap requires a volume column with data")
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    if n is None:
        cum_vol = df["volume"].cumsum()
        return pv.cumsum() / cum_vol.replace(0.0, np.nan)
    if n < 1:
        raise ValueError("n must be >= 1")
    num = pv.rolling(window=n, min_periods=n).sum()
    den = df["volume"].rolling(window=n, min_periods=n).sum()
    return num / den.replace(0.0, np.nan)
