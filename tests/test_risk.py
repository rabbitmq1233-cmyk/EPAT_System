import unittest

import numpy as np
import pandas as pd

from epat.risk import (
    atr_position_size,
    atr_stop_levels,
    fixed_fraction_size,
    fractional_kelly,
    kelly_binary,
    kelly_continuous,
    trailing_stop,
    volatility_target,
)


class TestSizing(unittest.TestCase):
    def test_kelly_even_odds(self):
        # p=0.6, even money -> f = 2p - 1 = 0.2
        self.assertAlmostEqual(kelly_binary(0.6, 1.0), 0.2, places=10)

    def test_kelly_no_edge_is_zero(self):
        self.assertEqual(kelly_binary(0.5, 1.0), 0.0)

    def test_kelly_with_odds(self):
        # p=0.5, win 2 per 1 risked -> f = (0.5*2 - 0.5)/2 = 0.25
        self.assertAlmostEqual(kelly_binary(0.5, 2.0), 0.25, places=10)

    def test_kelly_continuous(self):
        self.assertAlmostEqual(kelly_continuous(0.1, 0.04), 2.5, places=10)

    def test_fractional_kelly(self):
        self.assertAlmostEqual(fractional_kelly(0.2, 0.5), 0.1, places=10)

    def test_fixed_fraction(self):
        self.assertAlmostEqual(fixed_fraction_size(100_000, 50, 0.25), 500.0)

    def test_vol_target_series(self):
        rng = np.random.default_rng(0)
        ret = pd.Series(rng.normal(0, 0.01, 300))
        lev = volatility_target(ret, target_vol=0.15, window=20)
        self.assertTrue((lev >= 0).all())
        self.assertTrue((lev <= 3.0).all())

    def test_atr_position_size_scales_with_risk(self):
        small = atr_position_size(100_000, 2.0, risk_per_trade=0.01, atr_multiplier=2.0)
        large = atr_position_size(100_000, 2.0, risk_per_trade=0.02, atr_multiplier=2.0)
        self.assertAlmostEqual(large, 2 * small)
        self.assertEqual(atr_position_size(100_000, 0.0), 0.0)


class TestStops(unittest.TestCase):
    def test_atr_stop_levels_long(self):
        stop, target = atr_stop_levels(100, 2.0, direction=1, sl_mult=2, tp_mult=3)
        self.assertAlmostEqual(stop, 96.0)
        self.assertAlmostEqual(target, 106.0)

    def test_atr_stop_levels_short(self):
        stop, target = atr_stop_levels(100, 2.0, direction=-1, sl_mult=2, tp_mult=3)
        self.assertAlmostEqual(stop, 104.0)
        self.assertAlmostEqual(target, 94.0)

    def test_trailing_stop_long(self):
        prices = pd.Series([100.0, 105.0, 103.0, 110.0])
        ts = trailing_stop(prices, direction=1, distance=3.0)
        self.assertAlmostEqual(float(ts.iloc[1]), 102.0)
        self.assertAlmostEqual(float(ts.iloc[3]), 107.0)


if __name__ == "__main__":
    unittest.main()
