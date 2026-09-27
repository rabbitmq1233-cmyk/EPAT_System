import unittest

import numpy as np
import pandas as pd

from epat.engine import (
    CostModel,
    EventConfig,
    extract_trades,
    run_event_driven,
    run_from_returns,
    run_vectorized,
)


def _ramp_ohlcv(n=200, start=100.0, step=0.5):
    close = start + np.arange(n) * step
    idx = pd.date_range("2020-01-01", periods=n, freq="B", name="date")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


class TestVectorized(unittest.TestCase):
    def test_long_matches_price_ratio(self):
        price = pd.Series([100.0, 110.0, 121.0])
        pos = pd.Series([1.0, 1.0, 1.0])
        res = run_vectorized(price, pos, initial_capital=100_000)
        self.assertAlmostEqual(res.metrics["total_return"], 0.21, places=8)
        self.assertAlmostEqual(float(res.equity.iloc[-1]), 121_000.0, places=4)

    def test_costs_reduce_return(self):
        price = pd.Series([100.0, 110.0, 121.0])
        pos = pd.Series([1.0, 1.0, 1.0])
        free = run_vectorized(price, pos, cost_model=CostModel(0, 0))
        costly = run_vectorized(price, pos, cost_model=CostModel(100, 0))
        self.assertLess(costly.metrics["total_return"], free.metrics["total_return"])

    def test_no_drawdown_on_monotonic(self):
        price = pd.Series(np.linspace(100, 200, 50))
        pos = pd.Series(1.0, index=price.index)
        res = run_vectorized(price, pos)
        self.assertAlmostEqual(res.metrics["max_drawdown"], 0.0, places=10)

    def test_flat_position_flat_equity(self):
        price = pd.Series([100.0, 90.0, 120.0, 80.0])
        pos = pd.Series([0.0, 0.0, 0.0, 0.0])
        res = run_vectorized(price, pos, initial_capital=50_000)
        self.assertAlmostEqual(float(res.equity.iloc[-1]), 50_000.0)

    def test_run_from_returns(self):
        returns = pd.Series([0.01, -0.02, 0.03, 0.0], index=pd.date_range("2020-01-01", periods=4))
        pos = pd.Series([1.0, -1.0, -1.0, 0.0], index=returns.index)
        res = run_from_returns(returns, pos, initial_capital=100_000)
        self.assertTrue(np.isfinite(res.metrics["total_return"]))


class TestEventDriven(unittest.TestCase):
    def test_long_profits_on_uptrend(self):
        df = _ramp_ohlcv()
        signal = pd.Series(1.0, index=df.index)
        res = run_event_driven(df, signal, cost_model=CostModel(0, 0), initial_capital=100_000)
        self.assertGreater(res.metrics["total_return"], 0)
        self.assertGreaterEqual(res.metrics["num_trades"], 1)

    def test_short_loses_on_uptrend(self):
        df = _ramp_ohlcv()
        signal = pd.Series(-1.0, index=df.index)
        res = run_event_driven(df, signal, cost_model=CostModel(0, 0), initial_capital=100_000)
        self.assertLess(res.metrics["total_return"], 0)

    def test_stops_cap_loss(self):
        df = _ramp_ohlcv()
        from epat.indicators import atr

        signal = pd.Series(1.0, index=df.index)
        cfg = EventConfig(use_stops=True, sl_mult=2.0, tp_mult=3.0)
        res = run_event_driven(df, signal, atr=atr(df, 14), cost_model=CostModel(0, 0), config=cfg)
        self.assertTrue(np.isfinite(res.metrics["total_return"]))

    def test_long_only_flag(self):
        df = _ramp_ohlcv()
        signal = pd.Series(-1.0, index=df.index)
        cfg = EventConfig(allow_short=False)
        res = run_event_driven(df, signal, config=cfg, initial_capital=100_000)
        self.assertAlmostEqual(res.metrics["num_trades"], 0.0)


class TestTradeExtraction(unittest.TestCase):
    def test_segments(self):
        price = pd.Series([100.0, 105.0, 110.0, 100.0, 90.0])
        pos = pd.Series([1.0, 1.0, 0.0, -1.0, -1.0])
        trades = extract_trades(pos, price)
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades.iloc[0]["direction"], 1)
        self.assertEqual(trades.iloc[1]["direction"], -1)


if __name__ == "__main__":
    unittest.main()
