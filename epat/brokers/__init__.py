"""Broker adapters: a common interface plus mock, Zerodha, Upstox and IB.

Nothing here is imported at package import time except the light-weight base
and mock broker, so the framework runs without any vendor SDK installed.
"""

from __future__ import annotations

from epat.brokers.base import (
    AuthRequiredError,
    Broker,
    BrokerError,
    BrokerNotAvailableError,
    Holding,
    Margin,
    Order,
    OrderRejected,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
    Tick,
    TransactionType,
    Validity,
)
from epat.brokers.execution import (
    ExecutionConfig,
    ExecutionResult,
    trade_signals,
)
from epat.brokers.ib import IBBroker
from epat.brokers.mock import MockBroker
from epat.brokers.registry import get_broker, list_brokers, register_broker
from epat.brokers.upstox import UpstoxBroker
from epat.brokers.zerodha import ZerodhaBroker

__all__ = [
    "Broker",
    "BrokerError",
    "BrokerNotAvailableError",
    "AuthRequiredError",
    "OrderRejected",
    "OrderRequest",
    "Order",
    "OrderStatus",
    "OrderType",
    "ProductType",
    "TransactionType",
    "Validity",
    "Position",
    "Holding",
    "Margin",
    "Tick",
    "MockBroker",
    "ZerodhaBroker",
    "UpstoxBroker",
    "IBBroker",
    "get_broker",
    "list_brokers",
    "register_broker",
    "ExecutionConfig",
    "ExecutionResult",
    "trade_signals",
]
