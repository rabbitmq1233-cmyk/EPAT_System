"""Execution bridge: turn a strategy signal series into broker orders.

``trade_signals`` replays a price series through any :class:`~epat.brokers.base.Broker`
implementation. With :class:`~epat.brokers.mock.MockBroker` this is a full
paper-trading run; with a live broker (Zerodha/Upstox/IB) ``mark_price`` is a
no-op and fills come from the venue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from epat.brokers.base import (
    Broker,
    BrokerError,
    Order,
    OrderRequest,
    OrderType,
    Position,
    ProductType,
    TransactionType,
)
from epat.config import TRADING_DAYS
from epat.metrics import summarize

ORDER_COLUMNS = [
    "order_id",
    "timestamp",
    "symbol",
    "transaction_type",
    "quantity",
    "order_type",
    "status",
    "price",
    "average_price",
    "filled_quantity",
    "message",
]
POSITION_COLUMNS = ["symbol", "quantity", "average_price", "last_price", "pnl", "value"]


@dataclass
class ExecutionConfig:
    """How the execution bridge sizes and routes orders."""

    size_fraction: float = 1.0
    quantity: int | None = None
    product: ProductType = ProductType.DELIVERY
    order_type: OrderType = OrderType.MARKET
    exchange: str = "NSE"
    instrument_token: str | None = None
    allow_short: bool = True
    tag: str | None = "epat"


@dataclass
class ExecutionResult:
    """Output of a broker execution run."""

    orders: pd.DataFrame
    equity: pd.Series
    positions: pd.DataFrame
    metrics: dict[str, float]
    name: str = "execution"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        keys = ("total_return", "sharpe", "num_orders", "realised_pnl")
        parts = ", ".join(f"{k}={self.metrics.get(k, float('nan')):.4g}" for k in keys)
        return f"ExecutionResult({parts})"


def _orders_frame(orders: list[Order]) -> pd.DataFrame:
    rows = [
        {
            "order_id": o.order_id,
            "timestamp": o.timestamp,
            "symbol": o.symbol,
            "transaction_type": o.transaction_type.value,
            "quantity": o.quantity,
            "order_type": o.order_type.value,
            "status": o.status.value,
            "price": o.price,
            "average_price": o.average_price,
            "filled_quantity": o.filled_quantity,
            "message": o.message,
        }
        for o in orders
    ]
    return pd.DataFrame(rows, columns=ORDER_COLUMNS)


def _positions_frame(positions: list[Position]) -> pd.DataFrame:
    rows = [
        {
            "symbol": p.symbol,
            "quantity": p.quantity,
            "average_price": p.average_price,
            "last_price": p.last_price,
            "pnl": p.pnl,
            "value": p.value,
        }
        for p in positions
    ]
    return pd.DataFrame(rows, columns=POSITION_COLUMNS)


def _safe_positions(broker: Broker) -> list[Position]:
    try:
        return broker.positions()
    except BrokerError:
        return []


def _equity(broker: Broker) -> float:
    method = getattr(broker, "equity", None)
    if callable(method):
        return float(method())
    try:
        return float(broker.margins().total)
    except BrokerError:
        return float("nan")


def trade_signals(
    broker: Broker,
    symbol: str,
    prices: pd.Series,
    signals: pd.Series,
    *,
    config: ExecutionConfig | None = None,
    periods_per_year: int = TRADING_DAYS,
    risk_free: float = 0.0,
) -> ExecutionResult:
    """Replay ``signals`` through ``broker`` for a single ``symbol``.

    Orders are placed only when the target position changes; sizing is a fixed
    fraction of current equity (or an explicit quantity).
    """
    price = pd.Series(prices, dtype="float64").dropna()
    if price.size < 2:
        raise ValueError("need at least two prices")
    signal = np.sign(pd.Series(signals, dtype="float64").reindex(price.index).fillna(0.0))
    cfg = config or ExecutionConfig()

    orders_log: list[Order] = []
    equity_curve: list[float] = []
    last_desired = 0

    for i, (_, px) in enumerate(price.items()):
        broker.mark_price(symbol, float(px))
        desired = int(signal.iloc[i - 1]) if i > 0 else 0
        if not cfg.allow_short and desired < 0:
            desired = 0

        # Trade only when the target direction changes (entry / exit / reversal),
        # not on every bar from equity drift.
        if desired != last_desired:
            current = int(
                sum(pos.quantity for pos in _safe_positions(broker) if pos.symbol == symbol)
            )
            if desired == 0:
                target = 0
            elif cfg.quantity is not None:
                target = cfg.quantity * desired
            else:
                equity_now = _equity(broker)
                if not np.isfinite(equity_now):
                    equity_now = 0.0
                target = int(cfg.size_fraction * equity_now / float(px)) * desired

            delta = target - current
            if delta != 0:
                request = OrderRequest(
                    symbol=symbol,
                    transaction_type=TransactionType.BUY if delta > 0 else TransactionType.SELL,
                    quantity=abs(delta),
                    order_type=cfg.order_type,
                    product=cfg.product,
                    exchange=cfg.exchange,
                    instrument_token=cfg.instrument_token,
                    tag=cfg.tag,
                )
                orders_log.append(broker.place_order(request))
            last_desired = desired

        equity_curve.append(_equity(broker))

    equity = pd.Series(equity_curve, index=price.index, name="equity")
    returns = equity.pct_change().fillna(0.0).rename("returns")
    metrics = summarize(equity, returns, None, risk_free=risk_free, periods_per_year=periods_per_year)
    metrics["num_orders"] = float(len(orders_log))
    metrics["realised_pnl"] = float(getattr(broker, "realised_pnl", float("nan")))

    return ExecutionResult(
        orders=_orders_frame(orders_log),
        equity=equity,
        positions=_positions_frame(_safe_positions(broker)),
        metrics=metrics,
        name=f"{broker.name}:{symbol}",
    )
