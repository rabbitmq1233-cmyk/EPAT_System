"""Event-driven backtester.

Bar-by-bar simulation closer to live trading: executes at bar close using the
previous bar's signal, supports long/short, fixed-fractional sizing and ATR
stop-loss / take-profit levels evaluated intrabar on high/low.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from epat.config import DEFAULT_RISK_FREE, TRADING_DAYS
from epat.data.schema import validate_ohlcv
from epat.engine.costs import CostModel
from epat.engine.result import BacktestResult
from epat.engine.trades import TRADE_COLUMNS
from epat.metrics import summarize
from epat.risk.stops import atr_stop_levels


@dataclass(frozen=True)
class EventConfig:
    """Configuration for the event-driven engine."""

    size_fraction: float = 1.0
    allow_short: bool = True
    use_stops: bool = False
    sl_mult: float = 2.0
    tp_mult: float = 3.0
    periods_per_year: int = TRADING_DAYS
    risk_free: float = DEFAULT_RISK_FREE


def run_event_driven(
    df: pd.DataFrame,
    signals: pd.Series,
    *,
    atr: pd.Series | None = None,
    cost_model: CostModel | None = None,
    config: EventConfig | None = None,
    initial_capital: float = 100_000.0,
    name: str = "event",
) -> BacktestResult:
    """Run an event-driven backtest.

    ``signals`` is a target-direction series (``+1`` long, ``-1`` short, ``0``
    flat). Optional ``atr`` enables ATR stop/target levels.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    data = validate_ohlcv(df)
    cfg = config or EventConfig()
    costs = cost_model or CostModel()

    sig = pd.Series(signals, dtype="float64").reindex(data.index).fillna(0.0)
    sig = np.sign(sig)
    atr_series = (
        pd.Series(atr, dtype="float64").reindex(data.index)
        if atr is not None
        else None
    )

    close = data["close"].to_numpy()
    high = data["high"].to_numpy()
    low = data["low"].to_numpy()
    index = data.index
    n = len(index)

    cash = float(initial_capital)
    units = 0.0  # signed: >0 long, <0 short
    direction = 0
    entry_price = 0.0
    entry_equity = 0.0
    stop = target = None

    equity_curve = np.empty(n, dtype="float64")
    dir_series = np.zeros(n, dtype="float64")
    records: list[dict] = []

    for i in range(n):
        desired = int(sig.iloc[i - 1]) if i > 0 else 0
        if not cfg.allow_short and desired < 0:
            desired = 0

        def _close_position(exit_price: float) -> None:
            nonlocal cash, units, direction, stop, target
            fill = costs.fill_price(exit_price, -direction)
            notional = units * fill
            cash += notional - costs.commission(notional)
            pnl = cash - entry_equity
            records.append(
                {
                    "entry_date": entry_date,
                    "exit_date": index[i],
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": float(fill),
                    "bars": i - entry_bar,
                    "ret": float(pnl / entry_equity) if entry_equity else 0.0,
                    "pnl": float(pnl),
                }
            )
            units = 0.0
            direction = 0
            stop = target = None

        # 1) intrabar stop / target check while a position is open
        if direction != 0 and cfg.use_stops and stop is not None and target is not None:
            if direction == 1:
                if low[i] <= stop:
                    _close_position(stop)
                elif high[i] >= target:
                    _close_position(target)
            else:
                if high[i] >= stop:
                    _close_position(stop)
                elif low[i] <= target:
                    _close_position(target)

        # 2) signal-driven exit / reversal at bar close
        if direction != 0 and desired != direction:
            _close_position(close[i])

        # 3) fresh entry
        if direction == 0 and desired != 0:
            fill = costs.fill_price(close[i], desired)
            if fill > 0:
                equity_before = cash + units * close[i]
                # Size so notional + commission fits within the equity budget.
                budget = cfg.size_fraction * equity_before
                units_candidate = budget / (fill * (1.0 + costs.commission_rate))
                if units_candidate > 0:
                    notional = units_candidate * fill
                    commission = costs.commission(notional)
                    units = units_candidate * desired
                    cash -= units * fill + commission
                    direction = desired
                    entry_price = float(fill)
                    entry_equity = cash + units * close[i]
                    entry_date = index[i]
                    entry_bar = i
                    if cfg.use_stops and atr_series is not None:
                        atr_val = float(atr_series.iloc[i])
                        if np.isfinite(atr_val) and atr_val > 0:
                            stop, target = atr_stop_levels(
                                entry_price, atr_val, direction,
                                cfg.sl_mult, cfg.tp_mult,
                            )
                    else:
                        stop = target = None

        equity_curve[i] = cash + units * close[i]
        dir_series[i] = direction

    # Mark any still-open position to market so it appears in the blotter.
    if direction != 0:
        mark_price = float(close[n - 1])
        pnl = (cash + units * mark_price) - entry_equity
        records.append(
            {
                "entry_date": entry_date,
                "exit_date": index[n - 1],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": mark_price,
                "bars": (n - 1) - entry_bar,
                "ret": float(pnl / entry_equity) if entry_equity else 0.0,
                "pnl": float(pnl),
            }
        )

    equity = pd.Series(equity_curve, index=index, name="equity")
    returns = equity.pct_change().fillna(0.0).rename("returns")
    positions = pd.Series(dir_series, index=index, name="position")
    trades = pd.DataFrame(records, columns=TRADE_COLUMNS)

    metrics = summarize(
        equity,
        returns,
        trades,
        risk_free=cfg.risk_free,
        periods_per_year=cfg.periods_per_year,
    )
    return BacktestResult(
        name=name,
        equity=equity,
        returns=returns,
        positions=positions,
        trades=trades,
        metrics=metrics,
    )
