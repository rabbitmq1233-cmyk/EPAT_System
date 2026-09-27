"""Machine learning: feature engineering, models, walk-forward CV, regimes."""

from __future__ import annotations

from epat.ml.features import build_lagged_features, make_labels
from epat.ml.models import LogisticRegression, StandardScaler
from epat.ml.regimes import GMMResult, fit_gmm
from epat.ml.strategy import ml_direction
from epat.ml.walk_forward import walk_forward_predict

__all__ = [
    "build_lagged_features",
    "make_labels",
    "StandardScaler",
    "LogisticRegression",
    "walk_forward_predict",
    "fit_gmm",
    "GMMResult",
    "ml_direction",
]
