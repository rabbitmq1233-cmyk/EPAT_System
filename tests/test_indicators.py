import unittest

import numpy as np
import pandas as pd

from epat.data import generate_ohlcv
from epat.indicators import (
    adf,
    atr,
    bollinger,
    cointegration,
    ema,
    half_life,
    ols,
    parkinson_vol,
    rsi,
    sma,
    true_range,
    vwap,
    zscore,
)


class TestTrendMomentum(unittest.TestCase):
    def test_sma_and_ema(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype="float64")
        self.assertAlmostEqual(float(sma(s, 2).iloc[-1]), 4.5)
        self.assertTrue(ema(s, 3).iloc[-1] > 0)

    def test_rsi_bounds(self):
        close = generate_ohlcv(n=400, seed=5)["close"]
        r = rsi(close).dropna()
        self.assertTrue((r >= 0).all() and (r <= 100).all())

    def test_rsi_all_up_is_100(self):
        rising = pd.Series(np.arange(1, 60, dtype="float64"))
        self.assertAlmostEqual(float(rsi(rising, 14).iloc[-1]), 100.0)


class TestVolatility(unittest.TestCase):
    def setUp(self):
        self.df = generate_ohlcv(n=400, seed=6)

    def test_true_range_and_atr_positive(self):
        tr = true_range(self.df).dropna()
        a = atr(self.df, 14).dropna()
        self.assertTrue((tr >= 0).all())
        self.assertTrue((a > 0).all())

    def test_bollinger_ordering(self):
        mid, up, low = bollinger(self.df["close"], 20, 2.0)
        valid = mid.notna()
        self.assertTrue((up[valid] >= mid[valid]).all())
        self.assertTrue((low[valid] <= mid[valid]).all())

    def test_parkinson_positive(self):
        pv = parkinson_vol(self.df, 20).dropna()
        self.assertTrue((pv > 0).all())


class TestVolume(unittest.TestCase):
    def test_vwap_within_range(self):
        df = generate_ohlcv(n=300, seed=7)
        v = vwap(df, 20).dropna()
        self.assertTrue((v > 0).all())


class TestStats(unittest.TestCase):
    def test_ols_recovers_line(self):
        x = np.arange(100, dtype="float64")
        y = 3.0 + 2.5 * x
        res = ols(y, x)
        self.assertAlmostEqual(res.beta, 2.5, places=6)
        self.assertAlmostEqual(res.alpha, 3.0, places=6)
        self.assertAlmostEqual(res.rsquared, 1.0, places=6)

    def test_adf_white_noise_stationary(self):
        rng = np.random.default_rng(1)
        wn = pd.Series(rng.normal(0, 1, 500))
        self.assertTrue(adf(wn).is_stationary)

    def test_adf_random_walk_not_stationary(self):
        rng = np.random.default_rng(2)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 600)))
        self.assertFalse(adf(rw).is_stationary)

    def test_half_life_ou(self):
        rng = np.random.default_rng(3)
        phi = 0.9
        eps = rng.normal(0, 1, 800)
        s = np.zeros(800)
        for i in range(1, 800):
            s[i] = phi * s[i - 1] + eps[i]
        hl = half_life(pd.Series(s))
        self.assertTrue(4.0 < hl < 9.0, msg=f"half-life {hl}")

    def test_half_life_no_reversion(self):
        rng = np.random.default_rng(4)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 400)))
        # A random walk has no practical mean reversion: any finite-sample
        # half-life must be very long relative to the sample.
        self.assertGreater(half_life(rw), 50.0)

    def test_cointegration_true_for_constructed_pair(self):
        rng = np.random.default_rng(5)
        x = np.cumsum(rng.normal(0, 1, 500))
        y = 2.0 * x + rng.normal(0, 0.5, 500)
        res = cointegration(pd.Series(y), pd.Series(x))
        self.assertTrue(res.is_cointegrated)
        self.assertAlmostEqual(res.hedge_ratio, 2.0, delta=0.1)

    def test_zscore_mean_zero(self):
        s = generate_ohlcv(n=300, seed=8)["close"]
        z = zscore(s, 20).dropna()
        self.assertLess(abs(float(z.mean())), 0.5)


if __name__ == "__main__":
    unittest.main()
