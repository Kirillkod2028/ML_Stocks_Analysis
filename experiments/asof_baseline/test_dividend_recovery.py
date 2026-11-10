"""Regression tests for reviewed event identity and calendar evidence."""
import copy
import json
import re
import unittest
from pathlib import Path

import pandas as pd

from apply_verified_dividends import validated_ledger, apply_to_panel
from collect_dividend_candidates import MONTHS


class DividendRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.here = Path(__file__).resolve().parent
        self.review = json.loads((self.here / "dividend_recovery/reviewed_events.json").read_text())
        self.calendar = pd.bdate_range("2021-01-01", "2026-11-06")
        # Explicit actual no-session dates relevant to these reviewed events.
        self.calendar = self.calendar.difference(pd.to_datetime([
            "2022-05-02", "2022-05-03", "2023-05-01", "2023-05-09", "2026-05-01",
        ]))

    def test_separate_same_date_dividends_are_both_counted(self):
        events, coverage = validated_ledger(self.review, self.calendar)
        panel = pd.DataFrame({"TRADEDATE": pd.to_datetime(["2022-12-16"]), "ticker": ["LKOH"],
                              "entry_date": pd.to_datetime(["2022-12-19"]),
                              "label_end": pd.to_datetime(["2022-12-23"]),
                              "entry_open": [5000.], "exit_close": [4207.]})
        result = apply_to_panel(panel, events, coverage)
        self.assertEqual(result.cash_dividends.iloc[0], 793.)
        self.assertAlmostEqual(result.total_return.iloc[0], 0.)

    def test_unreviewed_ticker_is_unknown(self):
        events, coverage = validated_ledger(self.review, self.calendar)
        panel = pd.DataFrame({"TRADEDATE": pd.to_datetime(["2024-01-01"]), "ticker": ["UNREVIEWED"],
                              "entry_date": pd.to_datetime(["2024-01-02"]),
                              "label_end": pd.to_datetime(["2024-01-08"]),
                              "entry_open": [100.], "exit_close": [101.]})
        result = apply_to_panel(panel, events, coverage)
        self.assertFalse(result.dividends_known.iloc[0])
        self.assertTrue(pd.isna(result.total_return.iloc[0]))

    def test_calendar_mismatch_requires_review(self):
        wrong = self.calendar.union(pd.DatetimeIndex(["2023-05-09"]))
        with self.assertRaisesRegex(ValueError, "disagrees with calendar"):
            validated_ledger(self.review, wrong)

    def test_missing_event_breaks_coverage_assertion(self):
        missing = copy.deepcopy(self.review)
        missing["events"].pop()
        with self.assertRaisesRegex(ValueError, "coverage count disagrees"):
            validated_ledger(missing, self.calendar)

    def test_sber_reconciles_independent_issuer_snapshot(self):
        source = json.loads((self.here / "dividend_recovery/snapshots/sber_history_table_facts.json").read_text())
        expected = sorted((pd.to_datetime(row["record_date"], dayfirst=True).strftime("%Y-%m-%d"), row["cash_amount"])
                          for row in source["rows"] if row["cash_amount"] > 0)
        actual = sorted((row["record_date"], row["cash_amount"])
                        for row in self.review["events"] if row["ticker"] == "SBER")
        self.assertEqual(actual, expected)
        self.assertTrue(any(r["payment_year"] == 2022 and r["cash_amount"] == 0 for r in source["rows"]))

    def test_lukoil_reconciles_independent_issuer_snapshot(self):
        source = json.loads((self.here / "dividend_recovery/snapshots/lukoil_history_table_facts.json").read_text())
        expected = []
        for row in source["rows"]:
            day, month, year = re.search(r"(\d{1,2})\s+(\w+)\s+(\d{4})", row["record_date_text"]).groups()
            record = f"{year}-{MONTHS[month]:02d}-{int(day):02d}"
            if "2021-01-01" <= record <= "2026-11-06":
                expected.append((record, row["cash_amount"]))
        actual = [(row["record_date"], row["cash_amount"])
                  for row in self.review["events"] if row["ticker"] == "LKOH"]
        self.assertEqual(sorted(actual), sorted(expected))

    def test_added_issuers_reconcile_primary_facts(self):
        for ticker in ("MOEX", "MTSS", "IRAO", "GAZP", "ROSN", "NVTK", "BSPB", "PHOR", "HYDR"):
            with self.subTest(ticker=ticker):
                source = json.loads((self.here / f"dividend_recovery/snapshots/{ticker.lower()}_history_table_facts.json").read_text())
                expected = sorted((row["record_date"], row["cash_amount"]) for row in source["rows"])
                actual = sorted((row["record_date"], row["cash_amount"])
                                for row in self.review["events"] if row["ticker"] == ticker)
                self.assertEqual(actual, expected)

    def test_mts_record_dates_do_not_copy_provider_ex_dates(self):
        events = {row["period"]: row for row in self.review["events"] if row["ticker"] == "MTSS"}
        # Finam's displayed record dates are wrong for these two events.
        self.assertEqual(events["1H2021"]["record_date"], "2021-10-12")
        self.assertEqual(events["1H2021"]["ex_date"], "2021-10-11")
        self.assertEqual(events["FY2021"]["record_date"], "2022-07-12")
        self.assertEqual(events["FY2021"]["ex_date"], "2022-07-11")

    def test_mts_and_moex_reconcile_independent_annual_totals(self):
        for ticker in ("MOEX", "MTSS"):
            source = json.loads((self.here / f"dividend_recovery/snapshots/{ticker.lower()}_history_table_facts.json").read_text())
            for year, expected in source["annual_totals"].items():
                actual = sum(e["cash_amount"] for e in self.review["events"]
                             if e["ticker"] == ticker and e["ex_date"].startswith(year))
                self.assertAlmostEqual(actual, expected, places=10)

    def test_gazprom_no_annual_dividend_does_not_erase_interim(self):
        events, coverage = validated_ledger(self.review, self.calendar)
        panel = pd.DataFrame({"TRADEDATE": pd.to_datetime(["2022-10-06"]), "ticker": ["GAZP"],
                              "entry_date": pd.to_datetime(["2022-10-07"]),
                              "label_end": pd.to_datetime(["2022-10-13"]),
                              "entry_open": [200.], "exit_close": [148.97]})
        result = apply_to_panel(panel, events, coverage)
        self.assertEqual(result.cash_dividends.iloc[0], 51.03)
        self.assertAlmostEqual(result.total_return.iloc[0], 0.)

    def test_irao_uses_issuer_precision(self):
        event = next(e for e in self.review["events"]
                     if e["ticker"] == "IRAO" and e["period"] == "FY2025")
        self.assertAlmostEqual(event["cash_amount"], 0.321425305784583, places=15)
        self.assertNotEqual(event["cash_amount"], 0.32143)

    def test_coverage_cannot_certify_future_absence(self):
        future = copy.deepcopy(self.review)
        future["coverage"][0]["end_date"] = "2030-01-01"
        with self.assertRaisesRegex(ValueError, "beyond review date"):
            validated_ledger(future, self.calendar)

    def test_reviewed_zero_event_interval_is_valid(self):
        review = copy.deepcopy(self.review)
        zero = dict(review["coverage"][0], ticker="ZERO", expected_event_count=0)
        review["coverage"] = [zero]
        review["events"] = []
        events, coverage = validated_ledger(review, self.calendar)
        panel = pd.DataFrame({"TRADEDATE": pd.to_datetime(["2024-01-01"]), "ticker": ["ZERO"],
                              "entry_date": pd.to_datetime(["2024-01-02"]),
                              "label_end": pd.to_datetime(["2024-01-08"]),
                              "entry_open": [100.], "exit_close": [101.]})
        result = apply_to_panel(panel, events, coverage)
        self.assertTrue(result.dividends_known.iloc[0])
        self.assertEqual(result.cash_dividends.iloc[0], 0.)

    def test_partial_history_does_not_fill_2026_with_zero(self):
        events, coverage = validated_ledger(self.review, self.calendar)
        panel = pd.DataFrame({"TRADEDATE": pd.to_datetime(["2026-01-12", "2026-01-12"]),
                              "ticker": ["PHOR", "HYDR"],
                              "entry_date": pd.to_datetime(["2026-01-13", "2026-01-13"]),
                              "label_end": pd.to_datetime(["2026-01-19", "2026-01-19"]),
                              "entry_open": [100., 100.], "exit_close": [101., 101.]})
        result = apply_to_panel(panel, events, coverage)
        self.assertFalse(result.dividends_known.any())
        self.assertTrue(result.total_return.isna().all())

    def test_phosagro_distinct_same_day_components(self):
        for record, components in (("2023-07-11", [48., 216.]), ("2024-07-11", [15., 294.])):
            actual = sorted(e["cash_amount"] for e in self.review["events"]
                            if e["ticker"] == "PHOR" and e["record_date"] == record)
            self.assertEqual(actual, components)

    def test_hydro_preserves_legal_weekend_record_dates(self):
        events = {e["period"]: e for e in self.review["events"] if e["ticker"] == "HYDR"}
        self.assertEqual(events["FY2020"]["record_date"], "2021-07-10")
        self.assertEqual(events["FY2021"]["record_date"], "2022-07-10")

    def test_coverage_rejects_null_and_reversed_dates(self):
        for change in ({"end_date": None}, {"start_date": "2026-11-07"}):
            bad = copy.deepcopy(self.review)
            bad["coverage"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validated_ledger(bad, self.calendar)

    def test_panel_rejects_wrong_share_basis_and_duplicate_keys(self):
        events, coverage = validated_ledger(self.review, self.calendar)
        panel = pd.DataFrame({"TRADEDATE": pd.to_datetime(["2024-01-01"]), "ticker": ["SBER"],
                              "entry_date": pd.to_datetime(["2024-01-02"]),
                              "label_end": pd.to_datetime(["2024-01-08"]),
                              "entry_open": [100.], "exit_close": [101.]})
        with self.assertRaisesRegex(ValueError, "price_basis"):
            apply_to_panel(panel.assign(price_basis="raw"), events, coverage)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            apply_to_panel(pd.concat([panel, panel]), events, coverage)


if __name__ == "__main__":
    unittest.main()
