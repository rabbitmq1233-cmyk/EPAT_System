import unittest

import pandas as pd

from epat.brokers import (
    BrokerError,
    ExecutionConfig,
    MockBroker,
    OrderRequest,
    OrderStatus,
    OrderType,
    ProductType,
    TransactionType,
    get_broker,
    list_brokers,
    trade_signals,
)
from epat.data import generate_ohlcv
from epat.strategies import ma_crossover


def _broker(**kwargs) -> MockBroker:
    broker = MockBroker(initial_cash=100_000, **kwargs)
    broker.mark_price("X", 100.0)
    return broker


class TestOrderRequest(unittest.TestCase):
    def test_rejects_bad_quantity(self):
        with self.assertRaises(ValueError):
            OrderRequest("X", TransactionType.BUY, 0)

    def test_limit_requires_price(self):
        with self.assertRaises(ValueError):
            OrderRequest("X", TransactionType.BUY, 10, order_type=OrderType.LIMIT)

    def test_stop_requires_trigger(self):
        with self.assertRaises(ValueError):
            OrderRequest("X", TransactionType.BUY, 10, order_type=OrderType.SL, price=95.0)


class TestMockBroker(unittest.TestCase):
    def test_market_buy_fills_and_moves_cash(self):
        broker = _broker()
        order = broker.place_order(OrderRequest("X", TransactionType.BUY, 100))
        self.assertEqual(order.status, OrderStatus.COMPLETE)
        self.assertEqual(order.filled_quantity, 100)
        self.assertAlmostEqual(broker.margins().available_cash, 100_000 - 100 * 100)

    def test_commission_and_slippage(self):
        broker = _broker(commission_bps=10, slippage_bps=10)
        order = broker.place_order(OrderRequest("X", TransactionType.BUY, 100))
        self.assertGreater(order.average_price, 100.0)  # slippage worsens the buy
        self.assertLess(broker.margins().available_cash, 100_000 - 100 * 100)

    def test_limit_order_waits_then_fills(self):
        broker = _broker()
        order = broker.place_order(
            OrderRequest("X", TransactionType.BUY, 10, order_type=OrderType.LIMIT, price=95.0)
        )
        self.assertEqual(order.status, OrderStatus.OPEN)
        broker.mark_price("X", 95.0)
        self.assertEqual(order.status, OrderStatus.COMPLETE)
        self.assertAlmostEqual(order.average_price, 95.0)

    def test_stop_order_triggers(self):
        broker = _broker()
        order = broker.place_order(
            OrderRequest(
                "X", TransactionType.SELL, 10,
                order_type=OrderType.SL, price=90.0, trigger_price=92.0,
            )
        )
        self.assertEqual(order.status, OrderStatus.TRIGGER_PENDING)
        broker.mark_price("X", 93.0)
        self.assertEqual(order.status, OrderStatus.TRIGGER_PENDING)
        broker.mark_price("X", 91.0)
        self.assertEqual(order.status, OrderStatus.COMPLETE)

    def test_cancel(self):
        broker = _broker()
        order = broker.place_order(
            OrderRequest("X", TransactionType.BUY, 10, order_type=OrderType.LIMIT, price=1.0)
        )
        self.assertTrue(order.is_open)
        self.assertEqual(broker.cancel_order(order.order_id).status, OrderStatus.CANCELLED)

    def test_reject_without_price(self):
        broker = MockBroker(initial_cash=10_000)
        order = broker.place_order(OrderRequest("NOPE", TransactionType.BUY, 1))
        self.assertEqual(order.status, OrderStatus.REJECTED)

    def test_positions_and_realised_pnl(self):
        broker = _broker()
        broker.place_order(OrderRequest("X", TransactionType.BUY, 100))
        broker.mark_price("X", 110.0)
        position = broker.positions()[0]
        self.assertEqual(position.quantity, 100)
        self.assertAlmostEqual(position.pnl, 1000.0)

        broker.place_order(OrderRequest("X", TransactionType.SELL, 100))
        self.assertAlmostEqual(broker.realised_pnl, 1000.0)
        self.assertEqual(broker.positions(), [])

    def test_holdings_only_delivery(self):
        broker = _broker()
        broker.place_order(OrderRequest("X", TransactionType.BUY, 5, product=ProductType.INTRADAY))
        self.assertEqual(broker.holdings(), [])
        broker.place_order(OrderRequest("X", TransactionType.BUY, 7, product=ProductType.DELIVERY))
        holdings = broker.holdings()
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0].quantity, 7)

    def test_streaming_callback(self):
        broker = _broker()
        ticks = []
        broker.subscribe("X", lambda t: ticks.append(t.last_price))
        broker.mark_price("X", 101.0)
        self.assertEqual(ticks, [101.0])


class TestRegistry(unittest.TestCase):
    def test_list_and_get(self):
        names = list_brokers()
        for expected in ("mock", "zerodha", "upstox", "ib"):
            self.assertIn(expected, names)
        self.assertIsInstance(get_broker("mock"), MockBroker)

    def test_unknown_broker(self):
        with self.assertRaises(BrokerError):
            get_broker("nope")


class TestExecutionBridge(unittest.TestCase):
    def setUp(self):
        self.df = generate_ohlcv(n=300, mu=0.15, sigma=0.2, regime=True, seed=7)
        self.signals = ma_crossover(self.df, fast=10, slow=30)

    def test_trades_only_on_signal_change(self):
        broker = MockBroker(initial_cash=100_000)
        broker.load_history("T", self.df)
        result = trade_signals(broker, "T", self.df["close"], self.signals)
        changes = int((self.signals.diff().fillna(self.signals.iloc[0]) != 0).sum())
        self.assertLessEqual(len(result.orders), changes + 1)
        self.assertGreater(len(result.orders), 0)

    def test_equity_and_metrics(self):
        broker = MockBroker(initial_cash=100_000)
        broker.load_history("T", self.df)
        result = trade_signals(broker, "T", self.df["close"], self.signals)
        self.assertEqual(len(result.equity), len(self.df))
        for key in ("total_return", "sharpe", "num_orders", "realised_pnl"):
            self.assertIn(key, result.metrics)
        self.assertEqual(result.name, "mock:T")

    def test_long_only_never_goes_short(self):
        broker = MockBroker(initial_cash=100_000)
        broker.load_history("T", self.df)
        result = trade_signals(
            broker, "T", self.df["close"], self.signals,
            config=ExecutionConfig(allow_short=False),
        )
        # SELL orders are exits to flat; total sold can never exceed total bought.
        buys = result.orders.loc[result.orders["transaction_type"] == "BUY", "quantity"].sum()
        sells = result.orders.loc[result.orders["transaction_type"] == "SELL", "quantity"].sum()
        self.assertLessEqual(float(sells), float(buys))
        if len(result.positions):
            self.assertTrue((result.positions["quantity"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
