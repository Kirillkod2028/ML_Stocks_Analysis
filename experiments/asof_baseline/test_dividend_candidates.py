import json
import unittest
from pathlib import Path

from collect_dividend_candidates import parse_snapshot


class CandidateParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = Path(__file__).resolve().parent / "dividend_recovery/bulk_probe/bcs_table_snapshots.json"
        cls.snapshots = {r["ticker"]: r for r in json.loads(p.read_text())}

    def test_same_date_distributions_remain_distinct(self):
        rows, _ = parse_snapshot(self.snapshots["CHMF"])
        same_date = [r for r in rows if r["record_date_candidate"] == "2024-06-18"]
        self.assertEqual(sorted(r["native_cash_amount_candidate"] for r in same_date), [38.3, 191.51])
        self.assertTrue(all("ex_date" not in r and "verified_at" not in r for r in rows))

    def test_future_calendar_entry_is_not_historical(self):
        rows, audit = parse_snapshot(self.snapshots["ROSN"])
        self.assertEqual(audit["historical_candidates"], 4)
        self.assertEqual(audit["future_candidates"], 1)
        self.assertEqual(rows[0]["status"], "future_unverified")

    def test_inaccessible_page_never_certifies_zero(self):
        rows, audit = parse_snapshot(self.snapshots["AGRO"])
        self.assertEqual(rows, [])
        self.assertFalse(audit["table_available"])
        self.assertFalse(audit["history_exhaustive_verified"])
        self.assertIn("403", audit["access_error"])


if __name__ == "__main__":
    unittest.main()
