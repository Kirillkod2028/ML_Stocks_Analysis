import json
import unittest

from collect_finam_candidates import DATA, collect, parse_row


class FinamCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = json.loads((DATA / "finam_snapshots.json").read_text())

    def test_distinct_same_date_payments_survive(self):
        rows, _, _ = collect(self.snapshot, "2021-01-01", "2026-11-10")
        values = rows.loc[(rows.ticker == "CHMF") &
                          (rows.record_date == "2024-06-18"), "cash_amount"].tolist()
        self.assertEqual(sorted(values), [38.3, 191.51])

    def test_redirect_and_missing_page_never_create_coverage(self):
        rows, quarantine, audit = collect(self.snapshot, "2021-01-01", "2026-11-10")
        self.assertFalse(rows.ticker.isin(["FIXP", "AGRO"]).any())
        self.assertEqual(set(quarantine.ticker), {"FIXP"})
        self.assertFalse(audit.complete_interval_certified.any())
        self.assertTrue(rows.ex_date.isna().all())
        self.assertTrue(rows.price_basis.eq("unknown").all())

    def test_currency_precision_and_date_semantics_preserved(self):
        row = parse_row("12.12.2024 | 13.12.2024 | 0,353137\u00a0₽ | 36,33%")
        self.assertEqual(row["cash_amount"], 0.353137)
        self.assertEqual(row["last_cum_date"], "2024-12-12")
        self.assertEqual(row["record_date"], "2024-12-13")
        self.assertEqual(row["currency"], "RUB")

    def test_future_calendar_row_is_excluded(self):
        probe = dict(next(p for p in self.snapshot["probes"] if p["ticker"] == "CHMF"))
        probe["rows"] = ["11.01.2027 | 12.01.2027 | 9,44 ₽ | 1%"]
        rows, quarantine, audit = collect({"probes": [probe]}, "2021-01-01", "2026-11-10")
        self.assertTrue(rows.empty)
        self.assertTrue(quarantine.empty)
        self.assertEqual(audit.window_rows.iloc[0], 0)
        self.assertFalse(audit.complete_interval_certified.iloc[0])


if __name__ == "__main__":
    unittest.main()
