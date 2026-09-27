"""Upstox API v2 adapter (REST, stdlib HTTP).

Auth flow
---------
1. ``login_url()`` -> user authorises, Upstox redirects back with ``code``.
2. ``generate_session(code)`` -> exchange for an ``access_token``.

Upstox identifies instruments by an *instrument key* (e.g.
``"NSE_EQ|INE002A01018"``) rather than a trading symbol, so pass it via
``OrderRequest.instrument_token`` and the first argument of ``historical``.
The ``transport`` argument is injectable for offline testing.
"""

from __future__ import annotations

import pandas as pd

from epat.brokers.base import (
    AuthRequiredError,
    Broker,
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
from epat.brokers.http import default_transport, request_json
from epat.data.schema import validate_ohlcv

DEFAULT_BASE_URL = "https://api.upstox.com/v2"

_ORDER_TYPE = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.SL: "SL",
    OrderType.SL_M: "SL-M",
}
_PRODUCT = {
    ProductType.DELIVERY: "D",
    ProductType.INTRADAY: "I",
    ProductType.COVER: "CO",
    ProductType.OCO: "OCO",
}
_PRODUCT_REVERSE = {
    "D": ProductType.DELIVERY,
    "I": ProductType.INTRADAY,
    "CO": ProductType.COVER,
    "OCO": ProductType.OCO,
}
_INTERVAL = {
    "1m": "1minute",
    "30m": "30minute",
    "1d": "day",
    "1wk": "week",
    "1mo": "month",
}
_STATUS = {
    "complete": OrderStatus.COMPLETE,
    "rejected": OrderStatus.REJECTED,
    "cancelled": OrderStatus.CANCELLED,
    "open": OrderStatus.OPEN,
    "open pending": OrderStatus.PENDING,
    "trigger pending": OrderStatus.TRIGGER_PENDING,
    "partially filled": OrderStatus.PARTIAL,
}


class UpstoxBroker(Broker):
    """Upstox v2 broker adapter."""

    name = "upstox"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        redirect_uri: str,
        access_token: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport=None,
    ) -> None:
        super().__init__()
        if not api_key or not api_secret or not redirect_uri:
            raise ValueError("api_key, api_secret and redirect_uri are required")
        self.api_key = api_key
        self.api_secret = api_secret
        self.redirect_uri = redirect_uri
        self.access_token = access_token
        self.base_url = base_url.rstrip("/")
        self._transport = transport or default_transport
        if access_token:
            self._connected = True

    # -- auth --------------------------------------------------------------
    def login_url(self) -> str:
        return (
            f"{self.base_url}/login/authorization/dialog"
            f"?response_type=code&client_id={self.api_key}"
            f"&redirect_uri={self.redirect_uri}"
        )

    def generate_session(self, code: str) -> str:
        payload = request_json(
            f"{self.base_url}/login/authorization/token",
            method="POST",
            headers={"Accept": "application/json"},
            form={
                "code": code,
                "client_id": self.api_key,
                "client_secret": self.api_secret,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code",
            },
            transport=self._transport,
        )
        self.access_token = payload.get("access_token")
        if not self.access_token:
            raise AuthRequiredError(f"Upstox token exchange failed: {payload}")
        self._connected = True
        return self.access_token

    def _headers(self) -> dict[str, str]:
        if not self.access_token:
            raise AuthRequiredError("no access token; call generate_session() first")
        return {"Accept": "application/json", "Authorization": f"Bearer {self.access_token}"}

    # -- data --------------------------------------------------------------
    def historical(
        self,
        symbol: str,
        start=None,
        end=None,
        interval: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """``symbol`` is the Upstox instrument key."""
        if interval not in _INTERVAL:
            raise ValueError(f"unsupported interval '{interval}'; supported {sorted(_INTERVAL)}")
        to_date = pd.Timestamp(end if end is not None else pd.Timestamp.now()).strftime("%Y-%m-%d")
        from_date = pd.Timestamp(start if start is not None else "2000-01-01").strftime("%Y-%m-%d")
        payload = request_json(
            f"{self.base_url}/historical-candle/{symbol}/{_INTERVAL[interval]}/{to_date}/{from_date}",
            headers=self._headers(),
            transport=self._transport,
        )
        candles = payload.get("data", {}).get("candles", [])
        if not candles:
            raise ValueError(f"no candles returned for {symbol} ({interval})")
        frame = pd.DataFrame(candles).iloc[:, :6]
        frame.columns = ["date", "open", "high", "low", "close", "volume"]
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None)
        frame = frame.dropna(subset=["date"]).set_index("date")
        return validate_ohlcv(frame)

    def ltp(self, instrument_keys) -> dict[str, float]:
        keys = [instrument_keys] if isinstance(instrument_keys, str) else list(instrument_keys)
        payload = request_json(
            f"{self.base_url}/market-quote/ltp",
            headers=self._headers(),
            params={"instrument_key": ",".join(keys)},
            transport=self._transport,
        )
        return {
            key: value.get("last_price")
            for key, value in payload.get("data", {}).items()
        }

    # -- trading -----------------------------------------------------------
    def place_order(self, request: OrderRequest) -> Order:
        if not request.instrument_token:
            raise ValueError("Upstox requires OrderRequest.instrument_token (instrument key)")
        form = {
            "quantity": request.quantity,
            "product": _PRODUCT[request.product],
            "validity": request.validity.value,
            "price": request.price or 0.0,
            "trigger_price": request.trigger_price or 0.0,
            "order_type": _ORDER_TYPE[request.order_type],
            "transaction_type": request.transaction_type.value,
            "instrument_token": request.instrument_token,
            "tag": request.tag,
            "is_amo": "false",
        }
        payload = request_json(
            f"{self.base_url}/order/place",
            method="POST",
            headers=self._headers(),
            form=form,
            transport=self._transport,
        )
        order_id = payload.get("data", {}).get("order_id", "")
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

    def cancel_order(self, order_id: str) -> Order:
        request_json(
            f"{self.base_url}/order/cancel",
            method="DELETE",
            headers=self._headers(),
            params={"order_id": order_id},
            transport=self._transport,
        )
        return Order(
            order_id=order_id,
            symbol="",
            transaction_type=TransactionType.BUY,
            quantity=0,
            order_type=OrderType.MARKET,
            product=ProductType.DELIVERY,
            status=OrderStatus.CANCELLED,
        )

    def orders(self) -> list[Order]:
        payload = request_json(
            f"{self.base_url}/order/retrieve-all",
            headers=self._headers(),
            transport=self._transport,
        )
        return [self._order_from_row(row) for row in payload.get("data", [])]

    def _order_from_row(self, row: dict) -> Order:
        status = _STATUS.get(str(row.get("status", "open")).lower(), OrderStatus.OPEN)
        return Order(
            order_id=str(row.get("order_id", "")),
            symbol=str(row.get("trading_symbol") or row.get("tradingsymbol") or ""),
            transaction_type=TransactionType(str(row.get("transaction_type", "BUY")).upper()),
            quantity=int(row.get("quantity") or 0),
            order_type=OrderType(str(row.get("order_type", "MARKET")).replace("SL-M", "SL_M").upper()),
            product=_PRODUCT_REVERSE.get(str(row.get("product", "D")), ProductType.DELIVERY),
            status=status,
            filled_quantity=int(row.get("filled_quantity") or 0),
            price=float(row.get("price") or 0.0),
            average_price=float(row.get("average_price") or 0.0),
            trigger_price=float(row.get("trigger_price") or 0.0),
            message=str(row.get("status_message") or ""),
        )

    def positions(self) -> list[Position]:
        payload = request_json(
            f"{self.base_url}/portfolio/short-term-positions",
            headers=self._headers(),
            transport=self._transport,
        )
        out: list[Position] = []
        for row in payload.get("data", []):
            qty = int(row.get("quantity") or 0)
            if qty == 0:
                continue
            out.append(
                Position(
                    symbol=str(row.get("trading_symbol") or row.get("tradingsymbol") or ""),
                    quantity=qty,
                    average_price=float(row.get("average_price") or 0.0),
                    last_price=float(row.get("last_price") or 0.0),
                    product=_PRODUCT_REVERSE.get(str(row.get("product", "I")), ProductType.INTRADAY),
                )
            )
        return out

    def holdings(self) -> list[Holding]:
        payload = request_json(
            f"{self.base_url}/portfolio/long-term-holdings",
            headers=self._headers(),
            transport=self._transport,
        )
        return [
            Holding(
                symbol=str(row.get("trading_symbol") or row.get("tradingsymbol") or ""),
                quantity=int(row.get("quantity") or 0),
                average_price=float(row.get("average_price") or 0.0),
                last_price=float(row.get("last_price") or 0.0),
            )
            for row in payload.get("data", [])
        ]

    def margins(self) -> Margin:
        payload = request_json(
            f"{self.base_url}/user/get-funds-and-margin",
            headers=self._headers(),
            params={"segment": "SEC"},
            transport=self._transport,
        )
        equity = payload.get("data", {}).get("equity", {})
        available = float(equity.get("available_margin") or 0.0)
        used = float(equity.get("used_margin") or 0.0)
        return Margin(available_cash=available, used_margin=used, total=available + used)
