"""Zerodha Kite Connect adapter (REST v3, stdlib HTTP).

Requires a Kite Connect app (api_key / api_secret). No vendor SDK is needed:
the adapter talks to the REST API directly via :mod:`epat.brokers.http`.

Auth flow
---------
1. ``login_url()`` -> send the user to Zerodha, who returns a ``request_token``.
2. ``generate_session(request_token)`` -> exchange it for an ``access_token``.

The ``transport`` argument is injectable so the request mapping can be tested
without network access.
"""

from __future__ import annotations

import hashlib
from io import StringIO

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
from epat.brokers.http import default_transport, request_json, request_text
from epat.data.schema import validate_ohlcv

DEFAULT_BASE_URL = "https://api.kite.trade"
LOGIN_URL = "https://kite.zerodha.com/connect/login"

_ORDER_TYPE = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.SL: "SL",
    OrderType.SL_M: "SL-M",
}
_PRODUCT = {
    ProductType.DELIVERY: "CNC",
    ProductType.INTRADAY: "MIS",
    ProductType.COVER: "CO",
    ProductType.OCO: "CNC",
}
_INTERVAL = {
    "1m": "minute",
    "3m": "3minute",
    "5m": "5minute",
    "10m": "10minute",
    "15m": "15minute",
    "30m": "30minute",
    "60m": "60minute",
    "1d": "day",
}
_STATUS = {
    "COMPLETE": OrderStatus.COMPLETE,
    "REJECTED": OrderStatus.REJECTED,
    "CANCELLED": OrderStatus.CANCELLED,
    "TRIGGER PENDING": OrderStatus.TRIGGER_PENDING,
    "OPEN": OrderStatus.OPEN,
    "MODIFY REJECTED": OrderStatus.REJECTED,
    "VALIDATION PENDING": OrderStatus.PENDING,
}


class ZerodhaBroker(Broker):
    """Zerodha Kite Connect broker adapter."""

    name = "zerodha"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport=None,
    ) -> None:
        super().__init__()
        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret are required")
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.base_url = base_url.rstrip("/")
        self._transport = transport or default_transport
        self._instruments: dict[str, pd.DataFrame] = {}
        if access_token:
            self._connected = True

    # -- auth --------------------------------------------------------------
    def login_url(self) -> str:
        return f"{LOGIN_URL}?api_key={self.api_key}&v=3"

    def generate_session(self, request_token: str) -> str:
        checksum = hashlib.sha256(
            f"{self.api_key}{request_token}{self.api_secret}".encode("utf-8")
        ).hexdigest()
        payload = request_json(
            f"{self.base_url}/session/token",
            method="POST",
            headers={"X-Kite-Version": "3"},
            form={
                "api_key": self.api_key,
                "request_token": request_token,
                "checksum": checksum,
            },
            transport=self._transport,
        )
        self.access_token = payload.get("data", {}).get("access_token")
        if not self.access_token:
            raise AuthRequiredError(f"Kite session exchange failed: {payload}")
        self._connected = True
        return self.access_token

    def _headers(self) -> dict[str, str]:
        if not self.access_token:
            raise AuthRequiredError("no access token; call generate_session() first")
        return {
            "X-Kite-Version": "3",
            "Authorization": f"token {self.api_key}:{self.access_token}",
        }

    # -- instruments / data ------------------------------------------------
    def instruments(self, exchange: str = "NSE") -> pd.DataFrame:
        """Return (and cache) the instrument dump for an exchange."""
        if exchange not in self._instruments:
            text = request_text(
                f"{self.base_url}/instruments/{exchange}",
                headers=self._headers(),
                transport=self._transport,
            )
            frame = pd.read_csv(StringIO(text), dtype=str)
            frame["instrument_token"] = pd.to_numeric(
                frame.get("instrument_token"), errors="coerce"
            )
            self._instruments[exchange] = frame
        return self._instruments[exchange]

    def instrument_token(self, symbol: str, exchange: str = "NSE") -> int:
        """Look up the numeric instrument token for a trading symbol."""
        dump = self.instruments(exchange)
        matches = dump.loc[dump["tradingsymbol"] == symbol, "instrument_token"]
        if matches.empty:
            raise KeyError(f"symbol '{symbol}' not found on {exchange}")
        return int(matches.iloc[0])

    def historical(
        self,
        symbol: str,
        start=None,
        end=None,
        interval: str = "1d",
        *,
        exchange: str = "NSE",
        continuous: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        if interval not in _INTERVAL:
            raise ValueError(
                f"unsupported interval '{interval}'; Zerodha supports {sorted(_INTERVAL)}"
            )
        token = self.instrument_token(symbol, exchange)
        params = {
            "from": pd.Timestamp(start).strftime("%Y-%m-%d") if start is not None else "2000-01-01",
            "to": pd.Timestamp(end).strftime("%Y-%m-%d") if end is not None else pd.Timestamp.now().strftime("%Y-%m-%d"),
            "continuous": int(bool(continuous)),
        }
        payload = request_json(
            f"{self.base_url}/instruments/historical/{token}/{_INTERVAL[interval]}",
            headers=self._headers(),
            params=params,
            transport=self._transport,
        )
        candles = payload.get("data", {}).get("candles", [])
        if not candles:
            raise ValueError(f"no candles returned for {symbol} ({interval})")
        frame = pd.DataFrame(
            candles, columns=["date", "open", "high", "low", "close", "volume"]
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None)
        frame = frame.dropna(subset=["date"]).set_index("date")
        return validate_ohlcv(frame)

    def ltp(self, symbols) -> dict[str, float]:
        names = [symbols] if isinstance(symbols, str) else list(symbols)
        instruments = "&".join(f"i={name}" for name in names)
        payload = request_json(
            f"{self.base_url}/quote/ltp?{instruments}",
            headers=self._headers(),
            transport=self._transport,
        )
        return {
            key.split(":", 1)[-1]: value.get("last_price")
            for key, value in payload.get("data", {}).items()
        }

    # -- trading -----------------------------------------------------------
    def place_order(self, request: OrderRequest) -> Order:
        form = {
            "tradingsymbol": request.symbol,
            "exchange": request.exchange,
            "transaction_type": request.transaction_type.value,
            "order_type": _ORDER_TYPE[request.order_type],
            "quantity": request.quantity,
            "product": _PRODUCT[request.product],
            "validity": request.validity.value,
            "price": request.price,
            "trigger_price": request.trigger_price,
            "tag": request.tag,
        }
        payload = request_json(
            f"{self.base_url}/orders/{request.variety}",
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

    def cancel_order(self, order_id: str, *, variety: str = "regular") -> Order:
        payload = request_json(
            f"{self.base_url}/orders/{variety}/{order_id}",
            method="DELETE",
            headers=self._headers(),
            transport=self._transport,
        )
        return self._order_from_row({"order_id": order_id, "status": "CANCELLED", "status_message": str(payload.get("data", ""))})

    def orders(self) -> list[Order]:
        payload = request_json(
            f"{self.base_url}/orders", headers=self._headers(), transport=self._transport
        )
        return [self._order_from_row(row) for row in payload.get("data", [])]

    def _order_from_row(self, row: dict) -> Order:
        return Order(
            order_id=str(row.get("order_id", "")),
            symbol=str(row.get("tradingsymbol", "")),
            transaction_type=TransactionType(str(row.get("transaction_type", "BUY")).upper()),
            quantity=int(row.get("quantity") or 0),
            order_type=OrderType(str(row.get("order_type", "MARKET")).replace("SL-M", "SL_M").upper()),
            product=ProductType(
                {"CNC": "DELIVERY", "MIS": "INTRADAY", "CO": "COVER", "NRML": "DELIVERY", "BO": "COVER"}.get(
                    str(row.get("product", "CNC")), "DELIVERY"
                )
            ),
            status=_STATUS.get(str(row.get("status", "OPEN")), OrderStatus.OPEN),
            filled_quantity=int(row.get("filled_quantity") or 0),
            price=float(row.get("price") or 0.0),
            average_price=float(row.get("average_price") or 0.0),
            trigger_price=float(row.get("trigger_price") or 0.0),
            exchange=str(row.get("exchange", "NSE")),
            message=str(row.get("status_message", "")),
        )

    def positions(self) -> list[Position]:
        payload = request_json(
            f"{self.base_url}/portfolio/positions",
            headers=self._headers(),
            transport=self._transport,
        )
        data = payload.get("data", {})
        rows = list(data.get("net", [])) + list(data.get("day", []))
        out: list[Position] = []
        for row in rows:
            qty = int(row.get("quantity") or 0)
            if qty == 0:
                continue
            out.append(
                Position(
                    symbol=str(row.get("tradingsymbol", "")),
                    quantity=qty,
                    average_price=float(row.get("average_price") or 0.0),
                    last_price=float(row.get("last_price") or 0.0),
                    exchange=str(row.get("exchange", "NSE")),
                )
            )
        return out

    def holdings(self) -> list[Holding]:
        payload = request_json(
            f"{self.base_url}/portfolio/holdings",
            headers=self._headers(),
            transport=self._transport,
        )
        return [
            Holding(
                symbol=str(row.get("tradingsymbol", "")),
                quantity=int(row.get("quantity") or 0),
                average_price=float(row.get("average_price") or 0.0),
                last_price=float(row.get("last_price") or 0.0),
            )
            for row in payload.get("data", [])
        ]

    def margins(self) -> Margin:
        payload = request_json(
            f"{self.base_url}/user/margins", headers=self._headers(), transport=self._transport
        )
        equity = payload.get("data", {}).get("equity", {})
        available = equity.get("available", {}).get("cash", 0.0)
        used = equity.get("utilised", {}).get("debits", 0.0)
        return Margin(
            available_cash=float(available),
            used_margin=float(used),
            total=float(available) + float(used),
        )
