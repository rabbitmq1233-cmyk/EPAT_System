"""Machine-learning direction strategy using walk-forward logistic regression."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from epat.ml.features import build_lagged_features
from epat.ml.models import LogisticRegression
from epat.ml.walk_forward import walk_forward_predict
from epat.strategies.base import close_series, register


@register("ml_direction")
def ml_direction(
    data,
    *,
    lags: int = 12,
    train_size: int = 250,
    test_size: int = 50,
    threshold: float = 0.0,
    long_only: bool = False,
    expanding: bool = False,
    model_factory: Callable[[], object] | None = None,
) -> pd.Series:
    """Predict next-bar direction with a walk-forward classifier.

    Features are lagged log returns; the label is the sign of the forward log
    return. Predictions map to a target position of ``+1`` (up) or ``-1`` (down).
    """
    close = close_series(data)
    X, y = build_lagged_features(close, lags=lags, horizon=1, threshold=threshold)
    if X.empty:
        return pd.Series(0.0, index=close.index, name="signal")

    factory = model_factory or (lambda: LogisticRegression())
    preds = walk_forward_predict(
        X, y, factory, train_size=train_size, test_size=test_size, expanding=expanding
    )

    signal = preds.replace({1.0: 1.0, 0.0: -1.0})
    if long_only:
        signal = signal.clip(lower=0.0)
    return signal.reindex(close.index).fillna(0.0).rename("signal")
