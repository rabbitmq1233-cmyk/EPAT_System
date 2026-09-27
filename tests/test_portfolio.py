import unittest

import numpy as np
import pandas as pd

from epat.portfolio import (
    correlation_matrix,
    efficient_frontier,
    equal_weights,
    kelly_allocation,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    random_portfolios,
)


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([0.10, 0.15])
        self.cov = np.array([[0.04, 0.006], [0.006, 0.09]])

    def test_equal_weights(self):
        w = equal_weights(4)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertTrue(np.allclose(w, 0.25))

    def test_portfolio_variance_manual(self):
        w = np.array([0.5, 0.5])
        expected = 0.25 * 0.04 + 0.25 * 0.09 + 2 * 0.25 * 0.006
        self.assertAlmostEqual(portfolio_variance(w, self.cov), expected, places=10)

    def test_portfolio_return(self):
        w = np.array([0.5, 0.5])
        self.assertAlmostEqual(portfolio_return(w, self.mu), 0.125, places=10)

    def test_min_variance_beats_equal(self):
        w_eq = equal_weights(2)
        w_mv = min_variance_weights(self.cov)
        self.assertLess(portfolio_volatility(w_mv, self.cov), portfolio_volatility(w_eq, self.cov))

    def test_max_sharpe_weights_sum_one(self):
        w = max_sharpe_weights(self.mu, self.cov, risk_free=0.02)
        self.assertAlmostEqual(w.sum(), 1.0, places=8)

    def test_kelly_allocation_finite(self):
        w = kelly_allocation(self.mu, self.cov)
        self.assertTrue(np.isfinite(w).all())

    def test_random_portfolios(self):
        df = random_portfolios(self.mu, self.cov, n_samples=500, seed=1)
        self.assertIn("sharpe", df.columns)
        self.assertTrue(np.allclose(df[["w0", "w1"]].sum(axis=1), 1.0))
        self.assertTrue((df["volatility"] >= 0).all())

    def test_efficient_frontier_monotone_vol(self):
        frontier = efficient_frontier(self.mu, self.cov, n_points=30)
        vols = frontier["volatility"].to_numpy()
        self.assertLessEqual(vols[0], vols[-1] + 1e-9)
        self.assertTrue(np.all(np.diff(vols) >= -1e-8))

    def test_correlation_matrix(self):
        rng = np.random.default_rng(0)
        returns = pd.DataFrame(rng.normal(0, 0.01, (200, 3)), columns=list("ABC"))
        corr = correlation_matrix(returns)
        self.assertEqual(corr.shape, (3, 3))
        self.assertTrue(np.allclose(np.diag(corr), 1.0))


if __name__ == "__main__":
    unittest.main()
