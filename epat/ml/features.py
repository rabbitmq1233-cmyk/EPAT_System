"""Feature and label construction for supervised learning."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_labels(
    close: pd.Series,
    horizon: int = 1,
    threshold: float = 0.0,
) -> pd.Series:
    """Binary label: 1 if the forward log return over ``horizon`` exceeds
    ``threshold``, else 0. The last ``horizon`` values are NaN."""
    ret = np.log(pd.Series(close, dtype="float64"))
    forward = ret.shift(-horizon) - ret
    return (forward > threshold).astype("float64").where(forward.notna())


def build_lagged_features(
    close: pd.Series,
    lags: int = 12,
    horizon: int = 1,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a lagged-log-return feature matrix and a direction label.

    Returns ``(X, y)`` aligned on the same index, with warm-up/tail NaNs dropped.
    """
    if lags < 1:
        raise ValueError("lags must be >= 1")
    series = pd.Series(close, dtype="float64")
    ret = np.log(series).diff()

    features = {f"lag_{i}": ret.shift(i) for i in range(1, lags + 1)}
    X = pd.DataFrame(features, index=series.index)
    y = make_labels(series, horizon=horizon, threshold=threshold).rename("target")

    joined = X.join(y).dropna()
    return joined[X.columns], joined["target"]
