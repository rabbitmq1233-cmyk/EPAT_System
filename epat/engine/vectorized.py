"""Vectorized backtester.

Fast, research-oriented: positions are expressed as a fraction of capital and
traded with a one-bar lag to avoid look-ahead. Costs are charged on turnover.
"""

from __future__ import annotations

import pandas as pd

from epat.config import DEFAULT_RISK_FREE, TRADING_DAYS
from epat.engine.costs import CostModel
from epat.engine.result import BacktestResult
from epat.engine.trades import extract_trades, extract_trades_from_returns
from epat.metrics import summarize


def net_returns(
    prices: pd.Series,
    positions: pd.Series,
    *,
    cost_model: CostModel | None = None,
) -> pd.Series:
    """Net-of-cost strategy return stream for a price series and positions.

    Positions are shifted one bar (no look-ahead) and costs are charged on
    turnover. Shared by :func:`run_vectorized` and the portfolio pipeline.
    """
    price = pd.Series(prices, dtype="float64").dropna()
    pos = pd.Series(positions, dtype="float64").reindex(price.index).fillna(0.0)
    costs = cost_model or CostModel()

    asset_ret = price.pct_change().fillna(0.0)
    effective = pos.shift(1).fillna(0.0)
    gross = effective * asset_ret

    turnover = effective.diff()
    if turnover.size:
        turnover.iloc[0] = effective.iloc[0]
    turnover = turnover.abs().fillna(0.0)
    cost = turnover * (costs.commission_rate + costs.slippage_rate)
    return (gross - cost).rename("returns")


def run_vectorized(
    prices: pd.Series,
    positions: pd.Series,
    *,
    cost_model: CostModel | None = None,
    initial_capital: float = 100_000.0,
    risk_free: float = DEFAULT_RISK_FREE,
    periods_per_year: int = TRADING_DAYS,
    name: str = "vectorized",
) -> BacktestResult:
    """Run a vectorized backtest.

    Parameters
    ----------
    prices:
        Close-price series.
    positions:
        Target position per bar, as a fraction of capital (e.g. ``1.0`` fully
        long, ``-1.0`` fully short, ``0`` flat). Signals are shifted one bar
        before being applied.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    price = pd.Series(prices, dtype="float64").dropna()
    if price.size < 2:
        raise ValueError("need at least two price observations")

    pos = pd.Series(positions, dtype="float64").reindex(price.index).fillna(0.0)
    costs = cost_model or CostModel()

    net = net_returns(price, pos, cost_model=costs)
    equity = (initial_capital * (1.0 + net).cumprod()).rename("equity")

    effective = pos.shift(1).fillna(0.0)
    trades = extract_trades(effective, price)
    metrics = summarize(
        equity,
        net,
        trades,
        risk_free=risk_free,
        periods_per_year=periods_per_year,
    )
    return BacktestResult(
        name=name,
        equity=equity,
        returns=net,
        positions=effective.rename("position"),
        trades=trades,
        metrics=metrics,
    )


def run_from_returns(
    strategy_returns: pd.Series,
    positions: pd.Series,
    *,
    cost_model: CostModel | None = None,
    initial_capital: float = 100_000.0,
    risk_free: float = DEFAULT_RISK_FREE,
    periods_per_year: int = TRADING_DAYS,
    name: str = "returns-stream",
) -> BacktestResult:
    """Backtest a pre-computed strategy-return stream (e.g. a spread or factor).

    Positions are shifted one bar and costs are charged on turnover, exactly as
    in :func:`run_vectorized`, but the return stream is supplied directly
    instead of being derived from a price series.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    stream = pd.Series(strategy_returns, dtype="float64").fillna(0.0)
    if stream.size < 2:
        raise ValueError("need at least two return observations")
    pos = pd.Series(positions, dtype="float64").reindex(stream.index).fillna(0.0)
    costs = cost_model or CostModel()

    effective = pos.shift(1).fillna(0.0)
    gross = effective * stream

    turnover = effective.diff()
    if turnover.size:
        turnover.iloc[0] = effective.iloc[0]
    turnover = turnover.abs().fillna(0.0)
    cost = turnover * (costs.commission_rate + costs.slippage_rate)

    net = (gross - cost).rename("returns")
    equity = (initial_capital * (1.0 + net).cumprod()).rename("equity")

    trades = extract_trades_from_returns(effective, net)
    metrics = summarize(
        equity,
        net,
        trades,
        risk_free=risk_free,
        periods_per_year=periods_per_year,
    )
    return BacktestResult(
        name=name,
        equity=equity,
        returns=net,
        positions=effective.rename("position"),
        trades=trades,
        metrics=metrics,
    )
