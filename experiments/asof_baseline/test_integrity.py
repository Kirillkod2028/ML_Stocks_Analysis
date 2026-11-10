"""Focused checks for source failure and point-in-time feature invariance."""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ml_stocks_analysis import data_pipeline as source
from build_panel import make_targets, test_feature_causality_and_units


class IntegrityChecks(unittest.TestCase):
    def test_unknown_dividends_are_not_zero_payouts(self):
        async def no_table(*args, **kwargs):
            return {"description": {"columns": [], "data": []}}

        with patch.object(source, "_iss_get", no_table):
            with self.assertRaisesRegex(RuntimeError, "did not return a dividends table"):
                asyncio.run(source.fetch_dividends(None, None, "SBER"))

    def test_verified_ex_date_is_adjustment_boundary(self):
        raw = pd.DataFrame({"TRADEDATE": pd.to_datetime(["2024-07-10", "2024-07-11"]),
                            "OPEN": [100., 90.], "HIGH": [100., 90.],
                            "LOW": [100., 90.], "CLOSE": [100., 90.],
                            "WAPRICE": [100., 90.]})
        event = pd.DataFrame({"ex_date": pd.to_datetime(["2024-07-11"]),
                              "value": [10.]})
        out = source.backward_adjust(raw, event)
        self.assertEqual(out.ADJ_CLOSE.tolist(), [90., 90.])
        self.assertNotIn("dte_ex_days", source.add_calendar_features(raw, event))

    def test_partial_iss_page_is_an_error(self):
        # _paginate passes params as a positional argument, so an explicit
        # mock uses the fourth parameter to inspect the next cursor.
        async def page(session, sem, url, params):
            if params["start"] == 0:
                return {"history": {"columns": ["TRADEDATE"],
                                    "data": [["2024-01-01"]]},
                        "history.cursor": {"data": [[0, 2, 1]]}}
            return None

        with patch.object(source, "_iss_get", page):
            with self.assertRaisesRegex(RuntimeError, "failed at offset 1"):
                asyncio.run(source._paginate(None, None, "test", {}))

    def test_missing_cross_section_remains_missing(self):
        d = pd.Timestamp("2024-01-01")
        frame = pd.DataFrame({"TRADEDATE": [d] * 3,
                              "x": [np.nan, np.nan, np.nan]})
        self.assertTrue(source.normalize_cs(frame, ["x"]).x.isna().all())
        frame["x"] = [1.0, 1.0, np.nan]
        self.assertEqual(source.normalize_cs(frame, ["x"]).x.isna().sum(), 1)

    def test_rsi_and_units(self):
        rising = pd.Series(np.arange(40, dtype=float) + 1)
        self.assertEqual(source._wilder_rsi(rising).iloc[-1], 100.0)
        test_feature_causality_and_units()

    def test_reverse_split_uses_verified_factor(self):
        dates = pd.to_datetime(["2024-07-12", "2024-07-15"])
        raw = pd.DataFrame({"TRADEDATE": dates, "OPEN": [100., 500000.],
                            "HIGH": [100., 500000.], "LOW": [100., 500000.],
                            "CLOSE": [100., 500000.],
                            "WAPRICE": [100., 500000.],
                            "VOLUME": [100000., 20.]})
        adjusted, _ = source._detect_and_fix_splits(raw, pd.DataFrame(), "VTBR")
        self.assertEqual(adjusted.CLOSE.iloc[0], adjusted.CLOSE.iloc[1])
        self.assertEqual(adjusted.VOLUME.iloc[0], adjusted.VOLUME.iloc[1])

    def test_daily_rank_is_independent_of_future_label_availability(self):
        date = pd.Timestamp("2024-07-10")
        rows = pd.DataFrame({"TRADEDATE": [date] * 3, "ticker": ["A", "B", "C"],
                             "x": [1., 2., 3.], "target": [1., 2., np.nan]})
        rank_before = source.normalize_cs(rows, ["x"])
        rank_with_label = source.normalize_cs(rows.assign(target=3.), ["x"])
        self.assertEqual(rank_before.loc[1, "x"], rank_with_label.loc[1, "x"])
        self.assertEqual(rank_before.loc[1, "x"], 0.)

    def test_five_calendar_sessions_with_suspension(self):
        days = pd.bdate_range("2024-01-01", periods=45)
        close = 100 + np.arange(45, dtype=float) + np.sin(np.arange(45))
        stock = pd.DataFrame({"OPEN": close, "CLOSE": close,
                              "HIGH": close + 1, "LOW": close - 1}, index=days)
        stock.iloc[30:35, :] = np.nan
        target = make_targets(stock, stock)
        self.assertFalse(target.iloc[26:34].complete_labels.any())
        self.assertEqual(target.iloc[20].label_end, days[25])


if __name__ == "__main__":
    unittest.main()
