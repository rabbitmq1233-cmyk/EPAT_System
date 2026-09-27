import unittest

import numpy as np
import pandas as pd

from epat.metrics import (
    annual_volatility,
    cagr,
    calmar_ratio,
    drawdown,
    expectancy,
    hit_ratio,
    information_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    total_return,
)


class TestMetrics(unittest.TestCase):
    def test_total_return(self):
        eq = pd.Series([100.0, 110.0, 121.0])
        self.assertAlmostEqual(total_return(eq), 0.21, places=10)

    def test_cagr_known(self):
        # 252 bars doubling -> ~100% CAGR
        eq = pd.Series(np.linspace(100, 200, 253))
        self.assertAlmostEqual(cagr(eq, 252), 1.0, delta=0.02)

    def test_sharpe_sign(self):
        rng = np.random.default_rng(0)
        good = pd.Series(rng.normal(0.001, 0.01, 500))
        bad = pd.Series(rng.normal(-0.001, 0.01, 500))
        self.assertGreater(sharpe_ratio(good), 0)
        self.assertLess(sharpe_ratio(bad), 0)

    def test_sortino_ignores_upside(self):
        r = pd.Series([0.02, -0.01, 0.03, -0.005, 0.04])
        self.assertGreater(sortino_ratio(r, 0.0), 0)

    def test_max_drawdown_known(self):
        eq = pd.Series([100.0, 120.0, 90.0, 110.0])
        self.assertAlmostEqual(max_drawdown(eq), -0.25, places=10)

    def test_drawdown_duration(self):
        eq = pd.Series([100.0, 120.0, 90.0, 110.0])
        d = drawdown(eq)
        self.assertFalse(d.recovered)
        self.assertEqual(d.duration, 2)
        self.assertAlmostEqual(d.max_drawdown, -0.25)

    def test_drawdown_recovery(self):
        eq = pd.Series([100.0, 80.0, 130.0])
        d = drawdown(eq)
        self.assertTrue(d.recovered)
        self.assertAlmostEqual(d.max_drawdown, -0.20)

    def test_simple_metrics(self):
        trades = pd.DataFrame({"pnl": [1.0, -1.0, 1.0, 2.0]})
        self.assertAlmostEqual(hit_ratio(trades), 0.75)
        self.assertAlmostEqual(profit_factor(trades), 4.0)  # 4 profit / 1 loss
        self.assertAlmostEqual(expectancy(trades), 0.75)

    def test_information_ratio_identical_is_nan(self):
        r = pd.Series([0.01, 0.02, -0.01])
        self.assertTrue(np.isnan(information_ratio(r, r)))

    def test_calmar_and_vol(self):
        eq = pd.Series([100.0, 120.0, 90.0, 130.0, 125.0, 140.0])
        self.assertTrue(np.isfinite(calmar_ratio(eq, 252)))
        self.assertTrue(annual_volatility(eq.pct_change()) >= 0)

    def test_summarize_keys(self):
        eq = pd.Series(np.linspace(100, 130, 253))
        out = summarize(eq, eq.pct_change())
        for key in ("sharpe", "sortino", "max_drawdown", "calmar", "num_trades"):
            self.assertIn(key, out)


if __name__ == "__main__":
    unittest.main()
