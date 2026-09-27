"""Canonical OHLCV schema and validation helpers.

Every downstream module assumes a ``pandas.DataFrame`` indexed by a
``DatetimeIndex`` named ``date`` with lower-case columns
``open, high, low, close, volume``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

_PRICE_COLUMNS = ["open", "high", "low", "close"]
_DATE_ALIASES = ("date", "datetime", "timestamp", "time")


class DataSchemaError(ValueError):
    """Raised when input data does not conform to the OHLCV schema."""


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    require_volume: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    """Return a clean, standardised OHLCV frame.

    Parameters
    ----------
    df:
        Raw input. If the index is not a ``DatetimeIndex`` the function looks
        for a date-like column to use as the index.
    require_volume:
        When ``True`` a missing ``volume`` column raises. When ``False`` a
        missing volume column is created and filled with ``NaN``.
    copy:
        Copy the input before mutating (default) or operate in place.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataSchemaError("expected a pandas DataFrame")

    out = df.copy() if copy else df
    out.columns = [str(c).strip().lower() for c in out.columns]

    if not isinstance(out.index, pd.DatetimeIndex):
        if isinstance(out.index, pd.RangeIndex):
            moved = False
            for alias in _DATE_ALIASES:
                if alias in out.columns:
                    out.index = pd.to_datetime(out[alias])
                    out = out.drop(columns=[alias])
                    moved = True
                    break
            if not moved:
                raise DataSchemaError(
                    "no DatetimeIndex and no date column found "
                    f"(looked for {_DATE_ALIASES})"
                )
        else:
            out.index = pd.to_datetime(out.index)

    out.index.name = "date"

    missing = [c for c in _PRICE_COLUMNS if c not in out.columns]
    if missing:
        raise DataSchemaError(f"missing required price columns: {missing}")

    if "volume" not in out.columns:
        if require_volume:
            raise DataSchemaError("missing required 'volume' column")
        out["volume"] = np.nan

    out = out[OHLCV_COLUMNS]
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # De-duplicate timestamps, keep the last, and sort chronologically.
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def log_returns(close: pd.Series) -> pd.Series:
    """Continuously compounded (log) returns."""
    series = pd.Series(close, dtype="float64")
    return np.log(series).diff()


def simple_returns(close: pd.Series) -> pd.Series:
    """Arithmetic percentage returns."""
    series = pd.Series(close, dtype="float64")
    return series.pct_change()
