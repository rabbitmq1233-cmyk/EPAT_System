"""Lightweight ML models implemented on NumPy.

These avoid a hard scikit-learn dependency. If scikit-learn is installed you
can supply any estimator with ``fit``/``predict_proba`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    """Column-wise standardisation (mean 0, unit variance)."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x) -> "StandardScaler":
        arr = np.asarray(x, dtype="float64")
        self.mean_ = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, x) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("scaler must be fitted before transform")
        arr = np.asarray(x, dtype="float64")
        return (arr - self.mean_) / self.std_

    def fit_transform(self, x) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class LogisticRegression:
    """Binary logistic regression trained by gradient descent.

    Parameters
    ----------
    learning_rate:
        Gradient-descent step size.
    epochs:
        Number of full-batch passes.
    l2:
        L2 regularisation strength (excludes the intercept).
    standardize:
        Standardise features internally.
    """

    learning_rate: float = 0.1
    epochs: int = 500
    l2: float = 1e-3
    standardize: bool = True

    weights_: np.ndarray | None = None
    scaler_: StandardScaler | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _prepare(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype="float64")
        if self.standardize:
            if self.scaler_ is None:
                raise RuntimeError("call fit before predict")
            arr = self.scaler_.transform(arr)
        return arr

    def fit(self, x, y) -> "LogisticRegression":
        arr = np.asarray(x, dtype="float64")
        target = np.asarray(y, dtype="float64").ravel()
        if arr.ndim != 2:
            raise ValueError("features must be a 2D array")
        if self.standardize:
            self.scaler_ = StandardScaler().fit(arr)
            arr = self.scaler_.transform(arr)

        n, d = arr.shape
        design = np.column_stack([np.ones(n), arr])
        weights = np.zeros(d + 1, dtype="float64")
        reg = np.ones(d + 1)
        reg[0] = 0.0  # do not regularise the intercept

        for _ in range(self.epochs):
            probs = self._sigmoid(design @ weights)
            grad = design.T @ (probs - target) / n + self.l2 * reg * weights
            weights -= self.learning_rate * grad
        self.weights_ = weights
        return self

    def predict_proba(self, x) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("call fit before predict")
        arr = self._prepare(x)
        design = np.column_stack([np.ones(arr.shape[0]), arr])
        positive = self._sigmoid(design @ self.weights_)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, x) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def sklearn_available() -> bool:
    """Whether scikit-learn is importable in this environment."""
    try:  # pragma: no cover - environment dependent
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False
