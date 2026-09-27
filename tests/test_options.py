import unittest

import numpy as np
import pandas as pd

from epat.data import generate_ohlcv
from epat.options import (
    bs_price,
    butterfly_payoff,
    ewma_vol,
    garch11_fit,
    implied_vol,
    option_greeks,
    parity_gap,
    premium_stats,
    realised_vol,
    straddle_breakevens,
    straddle_payoff,
    strangle_payoff,
    synthetic_call,
    synthetic_put,
    variance_premium,
)


class TestBlackScholes(unittest.TestCase):
    def test_known_prices(self):
        call = bs_price(100, 100, 1.0, 0.05, 0.20, "call")
        put = bs_price(100, 100, 1.0, 0.05, 0.20, "put")
        self.assertAlmostEqual(call, 10.4506, delta=0.01)
        self.assertAlmostEqual(put, 5.5735, delta=0.01)

    def test_parity_exact(self):
        call = bs_price(100, 100, 1.0, 0.05, 0.20, "call")
        put = bs_price(100, 100, 1.0, 0.05, 0.20, "put")
        self.assertAlmostEqual(parity_gap(call, put, 100, 100, 1.0, 0.05), 0.0, places=8)

    def test_synthetic_positions(self):
        call = bs_price(100, 100, 1.0, 0.05, 0.20, "call")
        put = bs_price(100, 100, 1.0, 0.05, 0.20, "put")
        self.assertAlmostEqual(synthetic_call(put, 100, 100, 1.0, 0.05), call, places=8)
        self.assertAlmostEqual(synthetic_put(call, 100, 100, 1.0, 0.05), put, places=8)

    def test_expiry_intrinsic(self):
        self.assertAlmostEqual(bs_price(120, 100, 0.0, 0.0, 0.2, "call"), 20.0)
        self.assertAlmostEqual(bs_price(80, 100, 0.0, 0.0, 0.2, "put"), 20.0)

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            bs_price(100, 100, 1.0, 0.05, 0.2, "banana")


class TestGreeks(unittest.TestCase):
    def test_put_call_delta_bounds(self):
        g = option_greeks(100, 100, 1.0, 0.05, 0.2, "call")
        self.assertTrue(0 < g.delta < 1)
        self.assertGreater(g.gamma, 0)
        self.assertGreater(g.vega, 0)
        self.assertLess(g.theta, 0)

    def test_put_delta_negative(self):
        g = option_greeks(100, 100, 1.0, 0.05, 0.2, "put")
        self.assertTrue(-1 < g.delta < 0)


class TestImpliedVol(unittest.TestCase):
    def test_roundtrip(self):
        price = bs_price(100, 105, 0.5, 0.03, 0.35, "call")
        iv = implied_vol(price, 100, 105, 0.5, 0.03, "call")
        self.assertAlmostEqual(iv, 0.35, delta=1e-4)

    def test_below_intrinsic_returns_nan(self):
        self.assertTrue(np.isnan(implied_vol(0.0001, 200, 100, 1.0, 0.0, "call")))


class TestPayoffs(unittest.TestCase):
    def test_straddle(self):
        s = np.array([80.0, 100.0, 120.0])
        payoff = straddle_payoff(s, 100, 8)
        self.assertAlmostEqual(float(payoff[1]), -8.0)
        self.assertAlmostEqual(float(payoff[2]), 12.0)

    def test_straddle_breakevens(self):
        lo, hi = straddle_breakevens(100, 8)
        self.assertEqual((lo, hi), (92.0, 108.0))

    def test_strangle(self):
        s = np.array([80.0, 100.0, 120.0])
        payoff = strangle_payoff(s, 90, 110, 2)
        self.assertAlmostEqual(float(payoff[1]), -2.0)
        self.assertAlmostEqual(float(payoff[2]), 8.0)

    def test_butterfly_max_at_middle(self):
        s = np.array([90.0, 100.0, 110.0])
        payoff = butterfly_payoff(s, 90, 100, 110, 1.0)
        self.assertAlmostEqual(float(payoff[0]), -1.0)
        self.assertAlmostEqual(float(payoff[1]), 9.0)


class TestVolEstimators(unittest.TestCase):
    def setUp(self):
        self.returns = (
            generate_ohlcv(n=500, sigma=0.25, regime=True, seed=12)["close"].pct_change().dropna()
        )

    def test_ewma_positive(self):
        self.assertTrue(float(ewma_vol(self.returns).iloc[-1]) > 0)

    def test_garch_fit(self):
        g = garch11_fit(self.returns, refine=False)
        self.assertLess(g.alpha + g.beta, 1.0)
        self.assertGreater(g.alpha, 0.0)
        self.assertTrue(np.isfinite(g.forecast_vol) and g.forecast_vol > 0)


class TestVariancePremium(unittest.TestCase):
    def test_premium_ratio(self):
        rv = realised_vol(
            generate_ohlcv(n=400, seed=13)["close"].pct_change().dropna(), window=21
        )
        iv = (rv * 1.25).dropna()
        stats = premium_stats(iv, rv)
        self.assertGreater(stats["mean_premium"], 0)
        self.assertAlmostEqual(stats["premium_pct_of_iv"], 0.2, delta=0.02)
        self.assertAlmostEqual(stats["share_positive"], 1.0)

    def test_variance_premium_sign(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        iv = pd.Series([0.20, 0.22, 0.18, 0.25, 0.30], index=idx)
        rv = pd.Series([0.15, 0.20, 0.17, 0.20, 0.28], index=idx)
        vp = variance_premium(iv, rv)
        self.assertTrue((vp > 0).all())


if __name__ == "__main__":
    unittest.main()
