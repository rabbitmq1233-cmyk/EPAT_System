"""Backtesting engine: costs, results, and vectorized/event-driven runners."""

from __future__ import annotations

from epat.engine.costs import CostModel
from epat.engine.event_driven import EventConfig, run_event_driven
from epat.engine.result import BacktestResult
from epat.engine.trades import extract_trades, extract_trades_from_returns
from epat.engine.vectorized import net_returns, run_from_returns, run_vectorized

__all__ = [
    "CostModel",
    "BacktestResult",
    "EventConfig",
    "run_vectorized",
    "run_from_returns",
    "run_event_driven",
    "extract_trades",
    "extract_trades_from_returns",
    "net_returns",
]
