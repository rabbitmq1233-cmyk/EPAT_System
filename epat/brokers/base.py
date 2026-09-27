"""Broker abstraction: orders, positions and the ``Broker`` interface.

Concrete adapters live alongside this module (``mock``, ``zerodha``,
``upstox``, ``ib``). The interface is intentionally small and broker-agnostic:
research code (strategies, sizing, execution) talks to ``Broker`` and never to
a vendor SDK directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class BrokerError(RuntimeError):
    """Base class for broker errors."""


class BrokerNotAvailableError(BrokerError):
    """Raised when a broker SDK / dependency is not installed."""


class AuthRequiredError(BrokerError):
    """Raised when an operation needs an authenticated session."""


class OrderRejected(BrokerError):
    """Raised (or surfaced on an Order) when the broker rejects an order."""


class TransactionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class ProductType(str, Enum):
    """Product codes normalised across brokers."""

    INTRADAY = "INTRADAY"
    DELIVERY = "DELIVERY"
    COVER = "COVER"
    OCO = "OCO"


class Validity(str, Enum):
    DAY = "DAY"
    IOC = "IOC"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    TRIGGER_PENDING = "TRIGGER_PENDING"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class OrderRequest:
    """A broker-agnostic order instruction."""

    symbol: str
    transaction_type: TransactionType
    quantity: int
    order_type: OrderType = OrderType.MARKET
    product: ProductType = ProductType.DELIVERY
    price: float | None = None
    trigger_price: float | None = None
    validity: Validity = Validity.DAY
    exchange: str = "NSE"
    variety: str = "regular"
    tag: str | None = None
    #: Broker-native instrument id (Kite instrument token / Upstox instrument key).
    instrument_token: str | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type in (OrderType.LIMIT, OrderType.SL) and self.price is None:
            raise ValueError(f"price is required for {self.order_type.value} orders")
        if self.order_type in (OrderType.SL, OrderType.SL_M) and self.trigger_price is None:
            raise ValueError(f"trigger_price is required for {self.order_type.value} orders")


@dataclass
class Order:
    """A broker order and its current state."""

    order_id: str
    symbol: str
    transaction_type: TransactionType
    quantity: int
    order_type: OrderType
    product: ProductType
    status: OrderStatus
    filled_quantity: int = 0
    price: float = 0.0
    average_price: float = 0.0
    trigger_price: float = 0.0
    exchange: str = "NSE"
    timestamp: pd.Timestamp | None = None
    message: str = ""

    @property
    def is_open(self) -> bool:
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.TRIGGER_PENDING,
            OrderStatus.PARTIAL,
        )


@dataclass
class Position:
    symbol: str
    quantity: int
    average_price: float
    last_price: float
    product: ProductType = ProductType.DELIVERY
    exchange: str = "NSE"

    @property
    def pnl(self) -> float:
        return (self.last_price - self.average_price) * self.quantity

    @property
    def value(self) -> float:
        return self.last_price * self.quantity


@dataclass
class Holding:
    symbol: str
    quantity: int
    average_price: float
    last_price: float

    @property
    def pnl(self) -> float:
        return (self.last_price - self.average_price) * self.quantity


@dataclass
class Margin:
    available_cash: float
    used_margin: float
    total: float


@dataclass
class Tick:
    symbol: str
    last_price: float
    volume: float | None = None
    timestamp: pd.Timestamp | None = None


class Broker(ABC):
    """Abstract broker interface.

    Adapters implement the abstract members; everything else has a sensible
    default so partially-supported brokers still work.
    """

    name: str = "broker"

    def __init__(self) -> None:
        self._connected = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def login_url(self) -> str:
        """Return the vendor login URL (OAuth / connect login)."""
        raise BrokerError(f"{self.name} does not expose a login URL")

    def generate_session(self, request_token: str) -> str:
        """Exchange an auth code/request token for an access token."""
        raise BrokerError(f"{self.name} does not support token exchange")

    # -- market data -------------------------------------------------------
    @abstractmethod
    def historical(
        self,
        symbol: str,
        start=None,
        end=None,
        interval: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """Return a validated OHLCV frame for ``symbol``."""

    def ltp(self, symbols) -> dict[str, float]:
        """Last traded price for one or more symbols."""
        raise BrokerError(f"{self.name} does not provide LTP")

    def quote(self, symbols) -> dict[str, dict]:
        """Full quote for one or more symbols."""
        raise BrokerError(f"{self.name} does not provide quotes")

    def mark_price(self, symbol: str, price: float) -> None:
        """Update a local price cache. No-op for live brokers (paper/mock only)."""

    # -- trading -----------------------------------------------------------
    @abstractmethod
    def place_order(self, request: OrderRequest) -> Order:
        """Place an order and return its initial state."""

    def modify_order(self, order_id: str, **changes) -> Order:
        raise BrokerError(f"{self.name} does not support order modification")

    def cancel_order(self, order_id: str) -> Order:
        raise BrokerError(f"{self.name} does not support order cancellation")

    def orders(self) -> list[Order]:
        raise BrokerError(f"{self.name} does not expose the order book")

    def order_history(self, order_id: str) -> list[Order]:
        raise BrokerError(f"{self.name} does not expose order history")

    def positions(self) -> list[Position]:
        raise BrokerError(f"{self.name} does not expose positions")

    def holdings(self) -> list[Holding]:
        raise BrokerError(f"{self.name} does not expose holdings")

    def margins(self) -> Margin:
        raise BrokerError(f"{self.name} does not expose margins")

    # -- streaming ---------------------------------------------------------
    def subscribe(self, symbols, on_tick=None) -> None:
        raise BrokerError(f"{self.name} does not support streaming")

    def unsubscribe(self, symbols) -> None:
        raise BrokerError(f"{self.name} does not support streaming")
