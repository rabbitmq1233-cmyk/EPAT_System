"""Interactive Brokers adapter (optional ``ib_insync`` / ``ib_async``).

The vendor library is imported lazily: this module can always be imported, but
:meth:`IBBroker.connect` raises :class:`BrokerNotAvailableError` if no supported
library is installed. Pass ``ib=<client>`` to inject a client (used by tests).

Supported libraries (either works):
    pip install ib_insync     # or: pip install ib_async
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from epat.brokers.base import (
    Broker,
    BrokerNotAvailableError,
    Holding,
    Margin,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
    TransactionType,
)
from epat.data.schema import validate_ohlcv

_INSTALL_HINT = (
    "Interactive Brokers support needs an optional dependency: "
    "`pip install ib_insync` (or `pip install ib_async`)"
)


@dataclass
class IBSpec:
    """A vendor-neutral order specification for a TWS/IB order."""

    action: str
    quantity: int
    order_kind: str
    limit_price: float | None = None
    aux_price: float | None = None


def _load_ib_module():
    for name in ("ib_insync", "ib_async"):
        try:
            return __import__(name)
        except Exception:
            continue
    raise BrokerNotAvailableError(_INSTALL_HINT)


def order_spec(request: OrderRequest) -> IBSpec:
    """Map a broker-agnostic request to an IB order specification."""
    action = "BUY" if request.transaction_type == TransactionType.BUY else "SELL"
    if request.order_type == OrderType.MARKET:
        return IBSpec(action=action, quantity=request.quantity, order_kind="MarketOrder")
    if request.order_type == OrderType.LIMIT:
        return IBSpec(
            action=action, quantity=request.quantity, order_kind="LimitOrder",
            limit_price=request.price,
        )
    if request.order_type == OrderType.SL:
        return IBSpec(
            action=action, quantity=request.quantity, order_kind="StopLimitOrder",
            limit_price=request.price, aux_price=request.trigger_price,
        )
    return IBSpec(
        action=action, quantity=request.quantity, order_kind="StopOrder",
        aux_price=request.trigger_price,
    )


class IBBroker(Broker):
    """Interactive Brokers adapter over TWS / IB Gateway."""

    name = "ib"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        *,
        currency: str = "USD",
        exchange: str = "SMART",
        ib=None,
    ) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.currency = currency
        self.exchange = exchange
        self._ib = ib
        self._module = None

    # -- lifecycle ---------------------------------------------------------
    def connect(self):
        if self._ib is not None:
            self._connected = True
            return self._ib
        self._module = self._module or _load_ib_module()
        self._ib = self._module.IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id)
        self._connected = True
        return self._ib

    def disconnect(self) -> None:
        if self._ib is not None and hasattr(self._ib, "disconnect"):
            self._ib.disconnect()
        self._connected = False

    def _client(self):
        if self._ib is None:
            raise BrokerNotAvailableError("call connect() before using the IB broker")
        return self._ib

    # -- data --------------------------------------------------------------
    def contract(self, symbol: str, sec_type: str = "STK"):
        client = self._client()
        contract = client.Stock(symbol, self.exchange, self.currency)
        client.qualifyContracts(contract)
        return contract

    def historical(
        self,
        symbol: str,
        start=None,
        end=None,
        interval: str = "1d",
        *,
        duration: str = "5 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        **kwargs,
    ) -> pd.DataFrame:
        client = self._client()
        bars = client.reqHistoricalData(
            self.contract(symbol),
            endDateTime=end or "",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            raise ValueError(f"no historical bars returned for {symbol}")
        frame = pd.DataFrame(
            {
                "date": [bar.date for bar in bars],
                "open": [bar.open for bar in bars],
                "high": [bar.high for bar in bars],
                "low": [bar.low for bar in bars],
                "close": [bar.close for bar in bars],
                "volume": [bar.volume for bar in bars],
            }
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).set_index("date")
        return validate_ohlcv(frame)

    def ltp(self, symbols) -> dict[str, float]:
        client = self._client()
        names = [symbols] if isinstance(symbols, str) else list(symbols)
        out: dict[str, float] = {}
        for name in names:
            ticker = client.reqMktData(self.contract(name), "", False, False)
            client.sleep(0.5)
            out[name] = float(getattr(ticker, "last", float("nan")))
        return out

    # -- trading -----------------------------------------------------------
    def _ib_module(self):
        if self._module is not None:
            return self._module
        if self._ib is not None:
            top = type(self._ib).__module__.split(".")[0]
            try:
                self._module = __import__(top)
                return self._module
            except Exception:
                pass
        self._module = _load_ib_module()
        return self._module

    def _build_order(self, request: OrderRequest):
        module = self._ib_module()
        spec = order_spec(request)
        cls = getattr(module, spec.order_kind)
        kwargs = {"action": spec.action, "totalQuantity": spec.quantity}
        if spec.limit_price is not None:
            kwargs["lmtPrice"] = spec.limit_price
        if spec.aux_price is not None:
            kwargs["auxPrice"] = spec.aux_price
        return cls(**kwargs)

    def place_order(self, request: OrderRequest) -> Order:
        client = self._client()
        trade = client.placeOrder(self.contract(request.symbol), self._build_order(request))
        order = getattr(trade, "order", None)
        order_id = str(getattr(order, "orderId", "") or getattr(trade, "orderId", ""))
        return Order(
            order_id=order_id,
            symbol=request.symbol,
            transaction_type=request.transaction_type,
            quantity=request.quantity,
            order_type=request.order_type,
            product=request.product,
            status=OrderStatus.OPEN,
            price=request.price or 0.0,
            trigger_price=request.trigger_price or 0.0,
            exchange=request.exchange,
            timestamp=pd.Timestamp.now(),
        )

    def orders(self) -> list[Order]:
        client = self._client()
        trades = _call(client, "trades", [])
        out: list[Order] = []
        for trade in trades:
            order = getattr(trade, "order", None)
            if order is None:
                continue
            status_obj = getattr(trade, "orderStatus", None)
            action = str(getattr(order, "action", "BUY"))
            out.append(
                Order(
                    order_id=str(getattr(order, "orderId", "")),
                    symbol=str(getattr(getattr(trade, "contract", None), "symbol", "")),
                    transaction_type=(
                        TransactionType.BUY if action == "BUY" else TransactionType.SELL
                    ),
                    quantity=int(getattr(order, "totalQuantity", 0) or 0),
                    order_type=OrderType.MARKET,
                    product=ProductType.DELIVERY,
                    status=_map_status(str(getattr(status_obj, "status", "") or "")),
                    filled_quantity=int(getattr(status_obj, "filled", 0) or 0),
                    average_price=float(getattr(status_obj, "avgFillPrice", 0.0) or 0.0),
                )
            )
        return out

    def positions(self) -> list[Position]:
        client = self._client()
        out: list[Position] = []
        for item in _call(client, "positions", []):
            qty = int(getattr(item, "position", 0) or 0)
            if qty == 0:
                continue
            contract = getattr(item, "contract", None)
            avg_cost = float(getattr(item, "avgCost", 0.0) or 0.0)
            out.append(
                Position(
                    symbol=str(getattr(contract, "symbol", "")),
                    quantity=qty,
                    average_price=avg_cost,
                    last_price=avg_cost,
                )
            )
        return out

    def holdings(self) -> list[Holding]:
        return [
            Holding(
                symbol=pos.symbol,
                quantity=pos.quantity,
                average_price=pos.average_price,
                last_price=pos.last_price,
            )
            for pos in self.positions()
        ]

    def margins(self) -> Margin:
        client = self._client()
        values = {v.tag: v.value for v in _call(client, "accountValues", [])}
        total = float(values.get("NetLiquidation", 0.0))
        used = float(values.get("GrossPositionValue", 0.0))
        return Margin(available_cash=total - used, used_margin=used, total=total)


def _call(client, name: str, default):
    """Call a client method if it exists and is callable, else return the attribute."""
    attr = getattr(client, name, None)
    if callable(attr):
        return attr()
    return attr if attr is not None else default


def _map_status(status: str) -> OrderStatus:
    mapping = {
        "Filled": OrderStatus.COMPLETE,
        "Cancelled": OrderStatus.CANCELLED,
        "ApiCancelled": OrderStatus.CANCELLED,
        "Inactive": OrderStatus.REJECTED,
        "PreSubmitted": OrderStatus.PENDING,
        "Submitted": OrderStatus.OPEN,
        "PendingSubmit": OrderStatus.PENDING,
    }
    return mapping.get(status, OrderStatus.OPEN)
