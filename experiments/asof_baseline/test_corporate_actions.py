"""Focused tests for dividend entitlement, unknown data and split units."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ml_stocks_analysis.corporate_actions import (
    EVENT_COLUMNS, COVERAGE_COLUMNS, total_return_labels,
)


class CorporateActionTests(unittest.TestCase):
    def setUp(self):
        self.labels = pd.DataFrame([dict(ticker="AAA", entry_date="2024-01-08",
            exit_date="2024-01-12", entry_price=100., exit_price=90., price_basis="split_v1")])
        self.events = pd.DataFrame([dict(ticker="AAA", event_id="div1", ex_date="2024-01-10",
            cash_amount=10., currency="RUB", price_basis="split_v1",
            source_url="https://issuer.example/div1", verified_at="2026-11-10")])
        self.coverage = pd.DataFrame([dict(ticker="AAA", start_date="2024-01-01",
            end_date="2024-01-31", source_url="https://issuer.example/history",
            verified_at="2026-11-10")])

    def test_cash_receivable_offsets_ex_dividend_price_gap(self):
        r = total_return_labels(self.labels, self.events, self.coverage).iloc[0]
        self.assertAlmostEqual(r.price_return, -.1)
        self.assertAlmostEqual(r.total_return, 0.)

    def test_entry_on_ex_date_is_not_entitled_exit_on_ex_date_is(self):
        self.labels.loc[0, "entry_date"] = "2024-01-10"
        self.assertEqual(total_return_labels(self.labels, self.events, self.coverage).cash_dividends.iloc[0], 0.)
        self.labels.loc[0, "entry_date"] = "2024-01-08"
        self.labels.loc[0, "exit_date"] = "2024-01-10"
        self.assertEqual(total_return_labels(self.labels, self.events, self.coverage).cash_dividends.iloc[0], 10.)

    def test_missing_coverage_stays_unknown_even_with_some_events(self):
        empty = pd.DataFrame(columns=COVERAGE_COLUMNS)
        r = total_return_labels(self.labels, self.events, empty).iloc[0]
        self.assertFalse(r.dividends_known)
        self.assertTrue(pd.isna(r.total_return))

    def test_zero_requires_verified_complete_coverage(self):
        empty = pd.DataFrame(columns=EVENT_COLUMNS)
        r = total_return_labels(self.labels, empty, self.coverage).iloc[0]
        self.assertTrue(r.dividends_known)
        self.assertEqual(r.cash_dividends, 0.)

    def test_hole_in_coverage_is_unknown(self):
        a, b = self.coverage.copy(), self.coverage.copy()
        a.loc[0, "end_date"] = "2024-01-09"
        b.loc[0, "start_date"] = "2024-01-11"
        r = total_return_labels(self.labels, self.events, pd.concat([a, b])).iloc[0]
        self.assertFalse(r.dividends_known)
        b.loc[0, "start_date"] = "2024-01-10"
        r = total_return_labels(self.labels, self.events, pd.concat([a, b])).iloc[0]
        self.assertTrue(r.dividends_known)

    def test_split_unit_mismatch_raises(self):
        self.events.loc[0, "price_basis"] = "unadjusted_shares"
        with self.assertRaisesRegex(ValueError, "share units"):
            total_return_labels(self.labels, self.events, self.coverage)

    def test_duplicate_event_and_unsupported_currency_raise(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            total_return_labels(self.labels, pd.concat([self.events, self.events]), self.coverage)
        self.events.loc[0, "currency"] = "USD"
        with self.assertRaisesRegex(ValueError, "RUB"):
            total_return_labels(self.labels, self.events, self.coverage)


if __name__ == "__main__":
    unittest.main()
