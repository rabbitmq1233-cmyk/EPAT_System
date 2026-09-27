import os
import unittest

import pandas as pd

from epat.data.yahoo import (
    NIFTY,
    YahooError,
    build_url,
    fetch_yahoo,
    parse_chart,
)


def _payload():
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": "TEST",
                        "exchangeTimezoneName": "Asia/Kolkata",
                        "currency": "INR",
                    },
                    "timestamp": [1609459200, 1609545600, 1609632000, 1609718400],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0, None, 103.0],
                                "high": [102.0, 103.0, None, 105.0],
                                "low": [99.0, 100.0, None, 102.0],
                                "close": [101.0, 102.0, None, 104.0],
                                "volume": [1000.0, 1100.0, None, 1300.0],
                            }
                        ],
                        "adjclose": [{"adjclose": [101.0, 102.0, None, 104.0]}],
                    },
                }
            ],
            "error": None,
        }
    }


class TestParseChart(unittest.TestCase):
    def test_basic_shape_and_columns(self):
        df = parse_chart(_payload())
        self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(df.shape[0], 3)  # NaN-close row dropped
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.index.name, "date")
        self.assertTrue(df.index.is_monotonic_increasing)

    def test_daily_index_normalised(self):
        df = parse_chart(_payload(), interval="1d")
        self.assertTrue((df.index == df.index.normalize()).all())

    def test_intraday_index_not_normalised(self):
        df = parse_chart(_payload(), interval="60m")
        self.assertFalse((df.index == df.index.normalize()).all())

    def test_adjust_scales_prices(self):
        payload = _payload()
        # Half the close -> adjust must halve all price columns.
        payload["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"] = [
            50.5,
            51.0,
            None,
            52.0,
        ]
        df = parse_chart(payload, adjust=True)
        self.assertAlmostEqual(float(df["close"].iloc[0]), 50.5, places=6)
        self.assertAlmostEqual(float(df["open"].iloc[0]), 50.0, places=6)

    def test_no_adjust_keeps_raw(self):
        payload = _payload()
        payload["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"] = [
            50.5,
            51.0,
            None,
            52.0,
        ]
        df = parse_chart(payload, adjust=False)
        self.assertAlmostEqual(float(df["close"].iloc[0]), 101.0, places=6)

    def test_error_payload_raises(self):
        payload = {"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found"}}}
        with self.assertRaises(YahooError):
            parse_chart(payload)

    def test_empty_result_raises(self):
        with self.assertRaises(YahooError):
            parse_chart({"chart": {"result": [], "error": None}})

    def test_all_nan_rows_raise(self):
        payload = _payload()
        payload["chart"]["result"][0]["indicators"]["quote"][0]["close"] = [None, None, None, None]
        with self.assertRaises(YahooError):
            parse_chart(payload)


class TestBuildUrl(unittest.TestCase):
    def test_range_url(self):
        url = build_url("RELIANCE.NS", period="5y", interval="1d")
        self.assertIn("interval=1d", url)
        self.assertIn("range=5y", url)
        self.assertIn("RELIANCE.NS", url)

    def test_dates_url(self):
        url = build_url("AAPL", start="2020-01-01", end="2020-12-31", interval="1d")
        self.assertIn("period1=", url)
        self.assertIn("period2=", url)
        self.assertNotIn("range=", url)

    def test_start_only_still_sends_both_bounds(self):
        # Yahoo returns HTTP 400 unless both period1 and period2 are present.
        url = build_url("RELIANCE.NS", start="2018-01-01", interval="1d")
        self.assertIn("period1=", url)
        self.assertIn("period2=", url)
        self.assertNotIn("range=", url)

    def test_end_only_still_sends_both_bounds(self):
        url = build_url("RELIANCE.NS", end="2024-01-01", interval="1d")
        self.assertIn("period1=", url)
        self.assertIn("period2=", url)

    def test_caret_symbol_encoded(self):
        url = build_url("^NSEI")
        self.assertIn("%5ENSEI", url)

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            build_url("AAPL", interval="7d")

    def test_nifty_constant(self):
        self.assertEqual(NIFTY, "^NSEI")


@unittest.skipUnless(os.environ.get("EPAT_NETWORK_TESTS") == "1", "live network tests disabled")
class TestLiveFetch(unittest.TestCase):
    def test_fetch_aapl(self):
        df = fetch_yahoo("AAPL", period="5d", interval="1d")
        self.assertEqual(df.shape[1], 5)
        self.assertGreaterEqual(df.shape[0], 1)
        self.assertTrue((df["close"] > 0).all())

    def test_fetch_indian_symbol(self):
        df = fetch_yahoo("RELIANCE.NS", period="1mo", interval="1d")
        self.assertGreaterEqual(df.shape[0], 5)


if __name__ == "__main__":
    unittest.main()
