import unittest

import numpy as np
import pandas as pd

from epat.data import generate_multi
from epat.engine import CostModel, net_returns
from epat.portfolio import (
    apply_sizing,
    estimate_weight_history,
    run_portfolio_backtest,
)
from epat.strategies import ma_crossover


def _fixture(n=500, assets=3, seed=13):
    multi = generate_multi(n_assets=assets, n=n, seed=seed)
    prices = pd.DataFrame({k: v["close"] for k, v in multi.items()})
    signals = pd.DataFrame({k: ma_crossover(v, fast=10, slow=40) for k, v in multi.items()})
    return prices, signals


class TestApplySizing(unittest.TestCase):
    def setUp(self):
        self.multi = generate_multi(n_assets=1, n=300, seed=1)
        self.prices = next(iter(self.multi.values()))["close"]
        self.positions = pd.Series(1.0, index=self.prices.index)

    def test_none_is_identity(self):
        out = apply_sizing(self.prices, self.positions, method="none")
        pd.testing.assert_series_equal(out, self.positions, check_names=False)

    def test_fixed_scales(self):
        out = apply_sizing(self.prices, self.positions, method="fixed", fraction=0.5)
        self.assertTrue(np.allclose(out, 0.5))

    def test_vol_target_capped(self):
        out = apply_sizing(self.prices, self.positions, method="vol_target", target_vol=0.15, max_leverage=2.0, window=30)
        self.assertEqual(len(out), len(self.prices))
        self.assertTrue((out.abs() <= 2.0 + 1e-9).all())

    def test_kelly_non_negative(self):
        out = apply_sizing(self.prices, self.positions, method="kelly", window=30)
        self.assertTrue((out >= 0).all())

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            apply_sizing(self.prices, self.positions, method="banana")


class TestWeightHistory(unittest.TestCase):
    def test_no_lookahead_before_min_periods(self):
        prices, signals = _fixture()
        costs = CostModel(0, 0)
        net = pd.DataFrame(
            {c: net_returns(prices[c], signals[c], cost_model=costs) for c in prices.columns}
        )
        weights = estimate_weight_history(net, allocation="equal", min_periods=60)
        self.assertTrue((weights.iloc[:60].sum(axis=1) == 0).all())
        self.assertTrue((weights.iloc[60:].sum(axis=1) > 0).all())

    def test_equal_weights_sum_one(self):
        prices, signals = _fixture()
        net = pd.DataFrame({c: net_returns(prices[c], signals[c], cost_model=CostModel(0, 0)) for c in prices.columns})
        weights = estimate_weight_history(net, allocation="equal", min_periods=60)
        tail = weights.iloc[-1]
        self.assertAlmostEqual(float(tail.sum()), 1.0, places=8)


class TestPortfolioBacktest(unittest.TestCase):
    def test_basic_outputs(self):
        prices, signals = _fixture()
        result = run_portfolio_backtest(
            prices, signals, allocation="equal", allocation_window=60,
            rebalance=30, cost_model=CostModel(3, 1),
        )
        self.assertEqual(len(result.equity), len(prices))
        self.assertEqual(set(result.per_asset), set(prices.columns))
        for key in ("sharpe", "sortino", "max_drawdown", "num_assets"):
            self.assertIn(key, result.metrics)
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=6)

    def test_long_only_non_negative(self):
        prices, signals = _fixture()
        result = run_portfolio_backtest(
            prices, signals, allocation="max_sharpe", allocation_window=80,
            rebalance=40, long_only=True,
        )
        self.assertTrue((result.weights >= 0).all())

    def test_flat_before_allocation_window(self):
        prices, signals = _fixture(n=400)
        result = run_portfolio_backtest(prices, signals, allocation="equal", allocation_window=100)
        early = result.equity.iloc[:100]
        self.assertTrue(np.allclose(early, early.iloc[0]))

    def test_correlation_shape(self):
        prices, signals = _fixture(assets=4)
        result = run_portfolio_backtest(prices, signals, allocation="equal", allocation_window=60)
        self.assertEqual(result.correlation.shape, (4, 4))

    def test_unknown_allocation(self):
        prices, signals = _fixture(n=200)
        with self.assertRaises(ValueError):
            run_portfolio_backtest(prices, signals, allocation="nonsense", allocation_window=50)

    def test_sizing_wired(self):
        prices, signals = _fixture()
        fixed = run_portfolio_backtest(prices, signals, sizing="fixed", sizing_fraction=0.5, allocation="equal", allocation_window=60)
        full = run_portfolio_backtest(prices, signals, sizing="none", allocation="equal", allocation_window=60)
        self.assertNotAlmostEqual(
            fixed.metrics["annual_volatility"], full.metrics["annual_volatility"], places=6
        )


if __name__ == "__main__":
    unittest.main()
