"""In-memory paper-trading broker.

Deterministic, offline and dependency-free: fills market/limit/stop orders
against prices you feed it with :meth:`MockBroker.mark_price`, tracks cash,
positions, holdings and realised P&L. Used for paper trading, the execution
bridge, and the broker unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from epat.brokers.base import (
    Broker,
    Holding,
    Margin,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
    Tick,
    TransactionType,
)
from epat.data.schema import validate_ohlcv


@dataclass
class _Book:
    quantity: int = 0
    average_price: float = 0.0


class MockBroker(Broker):
    """A simulated broker for paper trading and tests."""

    name = "mock"

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        *,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        exchange: str = "NSE",
    ) -> None:
        super().__init__()
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.initial_cash = float(initial_cash)
        self._cash = float(initial_cash)
        self._commission_rate = commission_bps / 10_000.0
        self._slippage_rate = slippage_bps / 10_000.0
        self.exchange = exchange

        self._prices: dict[str, float] = {}
        self._orders: dict[str, Order] = {}
        self._book: dict[tuple[str, ProductType], _Book] = {}
        self._history: dict[str, pd.DataFrame] = {}
        self._subscribers: dict[str, list] = {}
        self._seq = 0
        self.realised_pnl = 0.0

    # -- data --------------------------------------------------------------
    def load_history(self, symbol: str, frame: pd.DataFrame) -> None:
        """Register historical OHLCV so :meth:`historical` can serve it."""
        clean = validate_ohlcv(frame)
        self._history[symbol] = clean
        self._prices.setdefault(symbol, float(clean["close"].iloc[-1]))

    def historical(self, symbol: str, start=None, end=None, interval: str = "1d", **kwargs) -> pd.DataFrame:
        if symbol not in self._history:
            raise KeyError(f"no history loaded for '{symbol}' (call load_history)")
        frame = self._history[symbol]
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.index <= pd.Timestamp(end)]
        return frame.copy()

    def ltp(self, symbols) -> dict[str, float]:
        names = [symbols] if isinstance(symbols, str) else list(symbols)
        return {name: self._prices[name] for name in names if name in self._prices}

    def quote(self, symbols) -> dict[str, dict]:
        return {
            name: {"last_price": price, "symbol": name}
            for name, price in self.ltp(symbols).items()
        }

    def mark_price(self, symbol: str, price: float) -> None:
        """Set the last price, fill eligible orders, and emit ticks."""
        self._prices[symbol] = float(price)
        for order in list(self._orders.values()):
            if order.symbol == symbol and order.is_open:
                self._try_fill(order)
        for callback in self._subscribers.get(symbol, []):
            callback(Tick(symbol=symbol, last_price=float(price), timestamp=pd.Timestamp.now()))

    # -- trading -----------------------------------------------------------
    def _next_id(self) -> str:
        self._seq += 1
        return f"MOCK{self._seq:05d}"

    def _reject(self, request: OrderRequest, message: str) -> Order:
        order = Order(
            order_id=self._next_id(),
            symbol=request.symbol,
            transaction_type=request.transaction_type,
            quantity=request.quantity,
            order_type=request.order_type,
            product=request.product,
            status=OrderStatus.REJECTED,
            price=request.price or 0.0,
            trigger_price=request.trigger_price or 0.0,
            exchange=request.exchange,
            timestamp=pd.Timestamp.now(),
            message=message,
        )
        self._orders[order.order_id] = order
        return order

    def place_order(self, request: OrderRequest) -> Order:
        if request.symbol not in self._prices:
            return self._reject(request, f"no market price for '{request.symbol}'")

        order = Order(
            order_id=self._next_id(),
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
        self._orders[order.order_id] = order

        if request.order_type == OrderType.MARKET:
            self._fill(order, self._prices[request.symbol])
        elif request.order_type == OrderType.LIMIT:
            self._try_fill(order)
        else:  # SL / SL-M -> wait for the trigger
            order.status = OrderStatus.TRIGGER_PENDING
        return order

    def modify_order(self, order_id: str, **changes) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"unknown order {order_id}")
        if not order.is_open:
            raise ValueError(f"cannot modify a {order.status.value} order")
        for field in ("price", "trigger_price", "quantity"):
            if field in changes and changes[field] is not None:
                setattr(order, field, changes[field])
        if order.status == OrderStatus.TRIGGER_PENDING:
            pass
        elif order.order_type == OrderType.LIMIT:
            self._try_fill(order)
        return order

    def cancel_order(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"unknown order {order_id}")
        if order.is_open:
            order.status = OrderStatus.CANCELLED
        return order

    def orders(self) -> list[Order]:
        return list(self._orders.values())

    def order_history(self, order_id: str) -> list[Order]:
        order = self._orders.get(order_id)
        return [order] if order is not None else []

    # -- fills -------------------------------------------------------------
    def _fills_now(self, order: Order, price: float) -> bool:
        if order.order_type == OrderType.MARKET:
            return True
        if order.order_type == OrderType.LIMIT:
            if order.transaction_type == TransactionType.BUY:
                return price <= order.price
            return price >= order.price
        # stop orders: buy stop above trigger, sell stop below trigger
        if order.transaction_type == TransactionType.BUY:
            return price >= order.trigger_price
        return price <= order.trigger_price

    def _try_fill(self, order: Order) -> bool:
        price = self._prices.get(order.symbol)
        if price is None or not self._fills_now(order, price):
            return False
        self._fill(order, price)
        return True

    def _fill(self, order: Order, market_price: float) -> None:
        direction = 1 if order.transaction_type == TransactionType.BUY else -1
        fill_price = market_price * (1.0 + direction * self._slippage_rate)
        notional = fill_price * order.quantity
        commission = abs(notional) * self._commission_rate

        if order.transaction_type == TransactionType.BUY:
            self._cash -= notional + commission
        else:
            self._cash += notional - commission

        self._apply_to_book(order.symbol, order.product, direction, order.quantity, fill_price)

        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.status = OrderStatus.COMPLETE
        order.message = "filled"

    def _apply_to_book(
        self,
        symbol: str,
        product: ProductType,
        direction: int,
        quantity: int,
        price: float,
    ) -> None:
        key = (symbol, product)
        book = self._book.setdefault(key, _Book())
        signed = direction * quantity
        old_qty = book.quantity
        new_qty = old_qty + signed

        if old_qty == 0 or (old_qty > 0) == (signed > 0):
            total_cost = abs(old_qty) * book.average_price + quantity * price
            book.average_price = total_cost / abs(new_qty) if new_qty else 0.0
        else:
            closed = min(quantity, abs(old_qty))
            if old_qty > 0:
                self.realised_pnl += (price - book.average_price) * closed
            else:
                self.realised_pnl += (book.average_price - price) * closed
            if new_qty == 0:
                book.average_price = 0.0
            elif (new_qty > 0) != (old_qty > 0):
                book.average_price = price
        book.quantity = new_qty

    # -- portfolio ---------------------------------------------------------
    def positions(self) -> list[Position]:
        out: list[Position] = []
        for (symbol, product), book in self._book.items():
            if book.quantity == 0:
                continue
            out.append(
                Position(
                    symbol=symbol,
                    quantity=book.quantity,
                    average_price=book.average_price,
                    last_price=self._prices.get(symbol, book.average_price),
                    product=product,
                    exchange=self.exchange,
                )
            )
        return out

    def holdings(self) -> list[Holding]:
        out: list[Holding] = []
        for (symbol, product), book in self._book.items():
            if product != ProductType.DELIVERY or book.quantity == 0:
                continue
            out.append(
                Holding(
                    symbol=symbol,
                    quantity=book.quantity,
                    average_price=book.average_price,
                    last_price=self._prices.get(symbol, book.average_price),
                )
            )
        return out

    def margins(self) -> Margin:
        notional = sum(abs(pos.value) for pos in self.positions())
        return Margin(
            available_cash=self._cash,
            used_margin=notional,
            total=self.equity(),
        )

    def equity(self) -> float:
        """Cash plus mark-to-market value of all positions."""
        return self._cash + sum(pos.value for pos in self.positions())

    def net_worth(self) -> float:
        return self.equity()

    # -- streaming ---------------------------------------------------------
    def subscribe(self, symbols, on_tick=None) -> None:
        names = [symbols] if isinstance(symbols, str) else list(symbols)
        for name in names:
            self._subscribers.setdefault(name, [])
            if on_tick is not None:
                self._subscribers[name].append(on_tick)

    def unsubscribe(self, symbols) -> None:
        names = [symbols] if isinstance(symbols, str) else list(symbols)
        for name in names:
            self._subscribers.pop(name, None)
