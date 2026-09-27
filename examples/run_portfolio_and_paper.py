"""Portfolio backtest + broker paper trading, end to end.

    python examples/run_portfolio_and_paper.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from epat.brokers import ExecutionConfig, MockBroker, list_brokers, trade_signals
from epat.data import generate_multi
from epat.engine import CostModel
from epat.portfolio import run_portfolio_backtest
from epat.reporting import tearsheet
from epat.strategies import ma_crossover


def main() -> None:
    panel = generate_multi(n_assets=4, n=750, seed=7)
    prices = pd.DataFrame({k: v["close"] for k, v in panel.items()})
    signals = pd.DataFrame({k: ma_crossover(v, fast=10, slow=40) for k, v in panel.items()})

    print("=== Portfolio backtest (max-Sharpe allocation, vol-target sizing) ===")
    result = run_portfolio_backtest(
        prices,
        signals,
        sizing="vol_target",
        target_vol=0.15,
        allocation="max_sharpe",
        allocation_window=80,
        rebalance=21,
        cost_model=CostModel(commission_bps=3, slippage_bps=1),
        name="demo-portfolio",
    )
    print(result)
    print("\nweights (latest):")
    print(result.weights.round(4).to_string())
    print()

    print("=== Broker paper trading (mock) ===")
    symbol = "SYN1"
    broker = MockBroker(initial_cash=500_000, commission_bps=3, slippage_bps=2)
    broker.load_history(symbol, panel[symbol])
    execution = trade_signals(
        broker,
        symbol,
        panel[symbol]["close"],
        signals[symbol],
        config=ExecutionConfig(size_fraction=0.5),
    )
    print(f"adapters available: {list_brokers()}")
    print(f"orders placed: {len(execution.orders)} | realised P&L: {execution.metrics['realised_pnl']:,.2f}")
    print(execution.orders.tail(5).to_string(index=False))
    print()
    print(tearsheet(execution))


if __name__ == "__main__":
    main()
