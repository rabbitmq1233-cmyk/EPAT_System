import hashlib
import json
import unittest

from epat.brokers import (
    BrokerNotAvailableError,
    OrderRequest,
    OrderType,
    ProductType,
    TransactionType,
)
from epat.brokers.ib import IBBroker, order_spec
from epat.brokers.upstox import UpstoxBroker
from epat.brokers.zerodha import ZerodhaBroker


class FakeTransport:
    """A fake HTTP transport: returns canned payloads and records requests."""

    def __init__(self, routes):
        # routes: list of (url_substring, status, payload)
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"method": method, "url": url, "headers": headers, "body": body})
        for needle, status, payload in self.routes:
            if needle in url:
                data = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
                return status, {}, data
        return 404, {}, b'{"error":"no route"}'

    def last_body(self) -> str:
        body = self.calls[-1]["body"]
        return body.decode("utf-8") if body else ""


class TestZerodha(unittest.TestCase):
    def _broker(self, transport, **kwargs):
        broker = ZerodhaBroker("key", "secret", "token", transport=transport, **kwargs)
        return broker

    def test_login_url(self):
        broker = self._broker(FakeTransport([]))
        self.assertIn("api_key=key", broker.login_url())
        self.assertIn("/connect/login", broker.login_url())

    def test_generate_session_checksum(self):
        transport = FakeTransport([("/session/token", 200, {"data": {"access_token": "tok123"}})])
        broker = self._broker(transport, **{})
        broker.access_token = None
        token = broker.generate_session("reqtok")
        self.assertEqual(token, "tok123")
        body = transport.last_body()
        expected = hashlib.sha256(b"key" + b"reqtok" + b"secret").hexdigest()
        self.assertIn(f"checksum={expected}", body)
        self.assertIn("request_token=reqtok", body)

    def test_place_order_mapping(self):
        transport = FakeTransport([("/orders/regular", 200, {"data": {"order_id": "OID1"}})])
        broker = self._broker(transport)
        order = broker.place_order(OrderRequest("RELIANCE", TransactionType.BUY, 10))
        self.assertEqual(order.order_id, "OID1")
        body = transport.last_body()
        self.assertIn("tradingsymbol=RELIANCE", body)
        self.assertIn("transaction_type=BUY", body)
        self.assertIn("order_type=MARKET", body)
        self.assertIn("product=CNC", body)
        self.assertIn("quantity=10", body)
        self.assertEqual(transport.calls[-1]["method"], "POST")
        self.assertIn("token key:token", transport.calls[-1]["headers"]["Authorization"])

    def test_intraday_product_maps_to_mis(self):
        transport = FakeTransport([("/orders/regular", 200, {"data": {"order_id": "x"}})])
        broker = self._broker(transport)
        broker.place_order(
            OrderRequest("INFY", TransactionType.SELL, 5, product=ProductType.INTRADAY)
        )
        self.assertIn("product=MIS", transport.last_body())

    def test_cancel_uses_delete(self):
        transport = FakeTransport([("/orders/regular/OID9", 200, {"data": "cancelled"})])
        broker = self._broker(transport)
        broker.cancel_order("OID9")
        self.assertEqual(transport.calls[-1]["method"], "DELETE")
        self.assertIn("/orders/regular/OID9", transport.calls[-1]["url"])

    def test_historical(self):
        csv = b"instrument_token,tradingsymbol,exchange\n408065,RELIANCE,NSE\n"
        candles = {
            "data": {
                "candles": [
                    [1577836800, 100.0, 101.0, 99.0, 100.5, 1000],
                    [1577923200, 100.5, 102.0, 100.0, 101.5, 1200],
                ]
            }
        }
        transport = FakeTransport(
            [("/instruments/historical/", 200, candles), ("/instruments/NSE", 200, csv)]
        )
        broker = self._broker(transport)
        frame = broker.historical("RELIANCE", "2020-01-01", "2020-01-10", "1d")
        self.assertEqual(list(frame.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(frame.shape[0], 2)
        self.assertIn("/instruments/historical/408065/day", transport.calls[-1]["url"])

    def test_orders_status_mapping(self):
        transport = FakeTransport(
            [("/orders", 200, {"data": [{"order_id": "a", "status": "COMPLETE", "filled_quantity": 10, "tradingsymbol": "X", "quantity": 10, "transaction_type": "BUY", "order_type": "MARKET", "product": "CNC", "average_price": 5.0}]})]
        )
        broker = self._broker(transport)
        orders = broker.orders()
        self.assertEqual(orders[0].status.value, "COMPLETE")
        self.assertEqual(orders[0].filled_quantity, 10)


class TestUpstox(unittest.TestCase):
    def _broker(self, transport):
        return UpstoxBroker("key", "secret", "https://cb.example/cb", "utok", transport=transport)

    def test_login_url(self):
        broker = self._broker(FakeTransport([]))
        url = broker.login_url()
        self.assertIn("client_id=key", url)
        self.assertIn("redirect_uri=https://cb.example/cb", url)

    def test_generate_session(self):
        transport = FakeTransport(
            [("/login/authorization/token", 200, {"access_token": "utok2"})]
        )
        broker = self._broker(transport)
        broker.access_token = None
        self.assertEqual(broker.generate_session("code123"), "utok2")
        self.assertIn("grant_type=authorization_code", transport.last_body())

    def test_place_order_requires_instrument_key(self):
        broker = self._broker(FakeTransport([]))
        with self.assertRaises(ValueError):
            broker.place_order(OrderRequest("X", TransactionType.BUY, 1))

    def test_place_order_mapping(self):
        transport = FakeTransport([("/order/place", 200, {"data": {"order_id": "U1"}})])
        broker = self._broker(transport)
        order = broker.place_order(
            OrderRequest(
                "RELIANCE", TransactionType.BUY, 3,
                instrument_token="NSE_EQ|INE002A01018",
            )
        )
        self.assertEqual(order.order_id, "U1")
        body = transport.last_body()
        self.assertIn("order_type=MARKET", body)
        self.assertIn("product=D", body)
        self.assertIn("transaction_type=BUY", body)
        self.assertIn("instrument_token=NSE_EQ", body)
        self.assertIn("Bearer utok", transport.calls[-1]["headers"]["Authorization"])

    def test_historical(self):
        candles = {
            "data": {
                "candles": [
                    ["2020-01-01T00:00:00+05:30", 100.0, 101.0, 99.0, 100.5, 1000, 0],
                    ["2020-01-02T00:00:00+05:30", 100.5, 102.0, 100.0, 101.5, 1200, 0],
                ]
            }
        }
        transport = FakeTransport([("/historical-candle/", 200, candles)])
        broker = self._broker(transport)
        frame = broker.historical("NSE_EQ|INE002A01018", "2020-01-01", "2020-01-10", "1d")
        self.assertEqual(frame.shape[0], 2)
        self.assertEqual(list(frame.columns), ["open", "high", "low", "close", "volume"])


class TestIB(unittest.TestCase):
    def test_order_spec_mapping(self):
        market = order_spec(OrderRequest("X", TransactionType.BUY, 10))
        self.assertEqual((market.action, market.order_kind), ("BUY", "MarketOrder"))

        limit = order_spec(
            OrderRequest("X", TransactionType.SELL, 5, order_type=OrderType.LIMIT, price=99.0)
        )
        self.assertEqual((limit.order_kind, limit.limit_price), ("LimitOrder", 99.0))

        stop_limit = order_spec(
            OrderRequest("X", TransactionType.BUY, 5, order_type=OrderType.SL, price=101.0, trigger_price=100.0)
        )
        self.assertEqual(stop_limit.order_kind, "StopLimitOrder")
        self.assertEqual(stop_limit.aux_price, 100.0)

        stop = order_spec(
            OrderRequest("X", TransactionType.SELL, 5, order_type=OrderType.SL_M, trigger_price=95.0)
        )
        self.assertEqual((stop.order_kind, stop.aux_price), ("StopOrder", 95.0))

    def test_connect_without_library(self):
        try:  # pragma: no cover - environment dependent
            import ib_insync  # noqa: F401
            available = True
        except Exception:
            try:
                import ib_async  # noqa: F401
                available = True
            except Exception:
                available = False
        if available:
            self.skipTest("IB library installed; BrokerNotAvailableError not expected")
        with self.assertRaises(BrokerNotAvailableError):
            IBBroker().connect()


if __name__ == "__main__":
    unittest.main()
