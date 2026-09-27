import unittest

import numpy as np
import pandas as pd

from epat.data import generate_ohlcv
from epat.ml import (
    LogisticRegression,
    StandardScaler,
    build_lagged_features,
    fit_gmm,
    ml_direction,
    walk_forward_predict,
)


class TestModels(unittest.TestCase):
    def test_standard_scaler(self):
        x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        scaled = StandardScaler().fit_transform(x)
        self.assertAlmostEqual(float(scaled.mean()), 0.0, places=8)
        self.assertAlmostEqual(float(scaled.std()), 1.0, places=8)

    def test_logistic_separable(self):
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(-1.5, 0.5, (200, 2)), rng.normal(1.5, 0.5, (200, 2))])
        y = np.array([0] * 200 + [1] * 200)
        model = LogisticRegression(epochs=400).fit(x, y)
        acc = float((model.predict(x) == y).mean())
        self.assertGreater(acc, 0.9)
        proba = model.predict_proba(x)
        self.assertAlmostEqual(float(proba.sum(axis=1).mean()), 1.0, places=6)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            LogisticRegression().predict(np.zeros((2, 2)))


class TestFeaturesAndWalkForward(unittest.TestCase):
    def test_feature_shapes(self):
        close = generate_ohlcv(n=300, seed=1)["close"]
        X, y = build_lagged_features(close, lags=5)
        self.assertEqual(X.shape[1], 5)
        self.assertEqual(len(X), len(y))
        self.assertTrue(set(np.unique(y.to_numpy())).issubset({0.0, 1.0}))

    def test_walk_forward_out_of_sample(self):
        close = generate_ohlcv(n=600, seed=2)["close"]
        X, y = build_lagged_features(close, lags=5)
        preds = walk_forward_predict(X, y, lambda: LogisticRegression(epochs=200), train_size=200, test_size=50)
        self.assertGreater(len(preds), 0)
        # first train_size rows must not be predicted (in-sample)
        self.assertTrue((preds.index >= X.index[200]).all())


class TestRegimes(unittest.TestCase):
    def test_gmm_recovers_two_means(self):
        rng = np.random.default_rng(3)
        data = np.concatenate([rng.normal(-2.0, 0.5, 500), rng.normal(2.0, 0.5, 500)])
        res = fit_gmm(data, n_components=2, seed=1)
        self.assertAlmostEqual(sorted(res.means)[0], -2.0, delta=0.3)
        self.assertAlmostEqual(sorted(res.means)[1], 2.0, delta=0.3)
        self.assertAlmostEqual(float(res.weights.sum()), 1.0, places=6)

    def test_gmm_regimes_on_returns(self):
        rng = np.random.default_rng(4)
        calm = rng.normal(0, 0.005, 600)
        stressed = rng.normal(0, 0.03, 400)
        res = fit_gmm(np.concatenate([calm, stressed]), n_components=2, seed=2)
        # variances must differ substantially between the two regimes
        self.assertGreater(max(res.variances) / min(res.variances), 3.0)


class TestMLStrategy(unittest.TestCase):
    def test_ml_direction_signal(self):
        df = generate_ohlcv(n=700, seed=5)
        sig = ml_direction(df, lags=5, train_size=250, test_size=50)
        self.assertEqual(len(sig), len(df))
        self.assertTrue(set(np.unique(sig.to_numpy())).issubset({-1.0, 0.0, 1.0}))

    def test_ml_long_only(self):
        df = generate_ohlcv(n=700, seed=6)
        sig = ml_direction(df, lags=5, train_size=250, test_size=50, long_only=True)
        self.assertTrue((sig >= 0).all())


if __name__ == "__main__":
    unittest.main()
