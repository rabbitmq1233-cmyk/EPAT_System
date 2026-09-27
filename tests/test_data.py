import unittest

import numpy as np
import pandas as pd

from epat.data import (
    DataSchemaError,
    generate_multi,
    generate_ohlcv,
    log_returns,
    save_csv,
    simple_returns,
    validate_ohlcv,
)


class TestSchema(unittest.TestCase):
    def test_validate_lowercases_and_sorts(self):
        idx = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
        df = pd.DataFrame(
            {"Open": [1, 2, 3], "High": [2, 3, 4], "Low": [0.5, 1, 1.5], "Close": [1.5, 2.5, 3.5]},
            index=idx,
        )
        out = validate_ohlcv(df)
        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "volume"])
        self.assertTrue(out.index.is_monotonic_increasing)
        self.assertEqual(out.index.name, "date")

    def test_missing_price_column_raises(self):
        df = pd.DataFrame({"open": [1], "high": [2], "close": [1.5]})
        with self.assertRaises(DataSchemaError):
            validate_ohlcv(df)

    def test_require_volume_raises(self):
        df = pd.DataFrame(
            {"open": [1], "high": [2], "low": [0.5], "close": [1.5]},
            index=pd.to_datetime(["2020-01-01"]),
        )
        with self.assertRaises(DataSchemaError):
            validate_ohlcv(df, require_volume=True)

    def test_log_and_simple_returns(self):
        close = pd.Series([100.0, 110.0, 121.0])
        self.assertAlmostEqual(float(log_returns(close).iloc[1]), np.log(1.1), places=10)
        self.assertAlmostEqual(float(simple_returns(close).iloc[1]), 0.1, places=10)


class TestSynthetic(unittest.TestCase):
    def test_generate_shape_and_determinism(self):
        a = generate_ohlcv(n=200, seed=1)
        b = generate_ohlcv(n=200, seed=1)
        self.assertEqual(a.shape, (200, 5))
        pd.testing.assert_frame_equal(a, b)

    def test_ohlc_consistency(self):
        df = generate_ohlcv(n=300, regime=True, seed=2)
        self.assertTrue((df["high"] >= df["low"]).all())
        self.assertTrue((df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all())
        self.assertTrue((df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all())
        self.assertTrue((df["close"] > 0).all())

    def test_generate_multi(self):
        multi = generate_multi(n_assets=3, n=150, seed=3)
        self.assertEqual(len(multi), 3)
        for frame in multi.values():
            self.assertEqual(frame.shape[0], 150)

    def test_csv_roundtrip(self):
        import tempfile
        from pathlib import Path

        df = generate_ohlcv(n=50, seed=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_csv(df, Path(tmp) / "x.csv")
            from epat.data import load_csv

            reloaded = load_csv(path)
            self.assertEqual(reloaded.shape[1], 5)
            self.assertAlmostEqual(float(reloaded["close"].iloc[-1]), float(df["close"].iloc[-1]), places=6)


if __name__ == "__main__":
    unittest.main()
