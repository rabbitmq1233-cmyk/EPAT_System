"""Walk-forward (out-of-sample) prediction."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], object],
    *,
    train_size: int = 250,
    test_size: int = 50,
    expanding: bool = False,
) -> pd.Series:
    """Generate out-of-sample predictions with a walk-forward split.

    Parameters
    ----------
    model_factory:
        Zero-argument callable returning a fresh estimator with
        ``fit(X, y)`` and ``predict(X)``.
    train_size, test_size:
        Number of observations in each train / test window.
    expanding:
        If ``True`` the training window grows (anchored at the first
        observation); otherwise it rolls forward a fixed length.

    Returns a Series of predictions indexed by the test observations.
    """
    feats = pd.DataFrame(X).dropna()
    target = pd.Series(y, dtype="float64").reindex(feats.index).dropna()
    feats = feats.loc[target.index]

    n = feats.shape[0]
    if train_size + 1 > n:
        raise ValueError("train_size larger than available data")

    preds = pd.Series(np.nan, index=feats.index, dtype="float64")
    start = 0
    while start + train_size < n:
        train_end = start + train_size
        test_end = min(train_end + test_size, n)

        train_slice = slice(0, train_end) if expanding else slice(start, train_end)
        model = model_factory()
        model.fit(feats.iloc[train_slice].to_numpy(), target.iloc[train_slice].to_numpy())

        test_x = feats.iloc[train_end:test_end].to_numpy()
        preds.iloc[train_end:test_end] = model.predict(test_x)

        if test_end >= n:
            break
        start += test_size

    return preds.dropna()
