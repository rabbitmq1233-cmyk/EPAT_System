import unittest

import numpy as np
import pandas as pd

from epat.data import generate_multi, generate_ohlcv
from epat.strategies import (
    bollinger_reversion,
    breakout_atr,
    ma_crossover,
    mean_reversion,
    momentum,
    pairs,
    pca_statarb,
    roll_return,
)


class TestSignalValidity(unittest.TestCase):
    def setUp(self):
        self.df = generate_ohlcv(n=600, regime=True, seed=9)

    def _assert_discrete(self, signal):
        self.assertEqual(len(signal), len(self.df))
        self.assertTrue(set(np.unique(signal.to_numpy())).issubset({-1.0, 0.0, 1.0}))

    def test_ma_crossover(self):
        self._assert_discrete(ma_crossover(self.df, fast=10, slow=40))

    def test_ma_long_only(self):
        sig = ma_crossover(self.df, fast=10, slow=40, long_only=True)
        self.assertTrue((sig >= 0).all())

    def test_breakout(self):
        self._assert_discrete(breakout_atr(self.df, entry_n=20, exit_n=10))

    def test_mean_reversion(self):
        self._assert_discrete(mean_reversion(self.df, window=20, entry_z=2.0))

    def test_bollinger(self):
        self._assert_discrete(bollinger_reversion(self.df, window=20, k=2.0))

    def test_momentum(self):
        self._assert_discrete(momentum(self.df, lookback=60))

    def test_momentum_vol_scaled(self):
        sig = momentum(self.df, lookback=60, volatility_scaled=True)
        self.assertTrue((sig.abs() <= 1.0 + 1e-9).all())

    def test_ma_bad_params(self):
        with self.assertRaises(ValueError):
            ma_crossover(self.df, fast=40, slow=10)


class TestPairsAndPCA(unittest.TestCase):
    def test_pairs(self):
        multi = generate_multi(n_assets=2, n=500, seed=10)
        keys = list(multi)
        res = pairs(multi[keys[0]]["close"], multi[keys[1]]["close"])
        self.assertTrue(np.isfinite(res.hedge_ratio))
        self.assertEqual(len(res.signal), len(res.spread))
        self.assertTrue(set(np.unique(res.signal.to_numpy())).issubset({-1.0, 0.0, 1.0}))

    def test_pca_statarb(self):
        multi = generate_multi(n_assets=4, n=500, seed=11)
        prices = pd.DataFrame({k: v["close"] for k, v in multi.items()})
        res = pca_statarb(prices, n_components=1, window=20)
        self.assertEqual(res.loadings.shape[1], 1)
        self.assertEqual(len(res.strategy_returns), len(prices) - 1)
        self.assertTrue(np.isfinite(res.strategy_returns.dropna()).all())


class TestRollReturn(unittest.TestCase):
    def test_contango_negative_roll(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        spot = pd.Series(100.0, index=idx)
        futures = pd.Series(np.linspace(100, 90, 100), index=idx)  # futures decay -> negative roll
        rr = roll_return(futures, spot).dropna()
        self.assertLess(float(rr.mean()), 0)

    def test_backwardation_positive_roll(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        spot = pd.Series(100.0, index=idx)
        futures = pd.Series(np.linspace(100, 110, 100), index=idx)  # futures rise -> positive roll
        rr = roll_return(futures, spot).dropna()
        self.assertGreater(float(rr.mean()), 0)


if __name__ == "__main__":
    unittest.main()
