import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from io import BytesIO
import json
import hashlib
from dataclasses import asdict
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ml_stocks_analysis.security_continuity import (
    bridge_held_security_quotes, validate_alias_history, VERIFIED_ALIASES,
    VERIFIED_TRANSITIONS, SAME_TICKER_TRANSITIONS, transition_records,
)
from recover_security_history import collect_identity, replay_verified_history


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.old = pd.DataFrame({"TRADEDATE": pd.to_datetime(["2025-02-14", "2025-02-17", "2025-02-18"]),
                                 "ticker": "AGRO", "OPEN": [np.nan]*3, "CLOSE": [np.nan]*3})
        self.new = pd.DataFrame({"TRADEDATE": pd.to_datetime(["2025-02-14", "2025-02-17", "2025-02-18"]),
                                 "ticker": "RAGR", "ISIN": "RU000A0JQUZ6", "OPEN": [99., 100., np.nan], "CLOSE": [99., 101., np.nan]})

    def test_no_implicit_conversion(self):
        result, audit = bridge_held_security_quotes(self.old, self.new, acknowledged_predecessors=set())
        self.assertTrue(result.CLOSE.isna().all())
        self.assertEqual(audit, [])

    def test_unit_conversion_preserves_pause_and_missing_quotes(self):
        result, _ = bridge_held_security_quotes(self.old, self.new, acknowledged_predecessors={"AGRO"})
        self.assertTrue(np.isnan(result.CLOSE.iloc[0]))
        self.assertAlmostEqual(result.CLOSE.iloc[1], 101*7.0135)
        self.assertTrue(np.isnan(result.CLOSE.iloc[2]))
        self.assertEqual(result.entry_allowed.tolist(), [True, False, False])

    def test_existing_entry_restriction_is_preserved(self):
        result, _ = bridge_held_security_quotes(self.old.assign(entry_allowed=False), self.new,
                                               acknowledged_predecessors={"AGRO"})
        self.assertFalse(result.entry_allowed.any())

    def test_missing_dates_fail_instead_of_silent_no_match(self):
        self.new.loc[0, "TRADEDATE"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "Missing quote date"):
            bridge_held_security_quotes(self.old, self.new, acknowledged_predecessors={"AGRO"})

    def test_alias_keeps_historical_code_and_excludes_old_gdr(self):
        alias = VERIFIED_ALIASES[0]
        history = pd.DataFrame({"TRADEDATE": ["2024-03-18", "2024-11-27"],
                                "ticker": "TCSG", "ISIN": alias.isin,
                                "OPEN": [3000., 2200.], "CLOSE": [3100., 2384.8]})
        result = validate_alias_history(history, alias)
        self.assertTrue(result.ticker.eq("TCSG").all())
        self.assertTrue(result.successor_ticker.eq("T").all())
        self.assertEqual(result.CLOSE.iloc[0], 3100.)
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_alias_history(history.assign(ISIN="US87238U2033"), alias)
        history.loc[0, "TRADEDATE"] = "2024-02-15"
        with self.assertRaisesRegex(ValueError, "legal-security interval"):
            validate_alias_history(history, alias)

    def test_wrong_security_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "ISIN"):
            bridge_held_security_quotes(self.old, self.new.assign(ISIN="WRONG"), acknowledged_predecessors={"AGRO"})

    def test_overlap_does_not_overwrite_original_price(self):
        self.old.loc[1, "CLOSE"] = 50.
        with self.assertRaisesRegex(ValueError, "overlap"):
            bridge_held_security_quotes(self.old, self.new, acknowledged_predecessors={"AGRO"})

    def test_future_successor_changes_do_not_rewrite_old_quotes(self):
        self.old.loc[0, "CLOSE"] = 1083.8
        result, _ = bridge_held_security_quotes(self.old, self.new, acknowledged_predecessors={"AGRO"})
        self.assertEqual(result.CLOSE.iloc[0], 1083.8)

    def test_metadata_identity_is_checked_independently(self):
        payload = {"description": {"columns": ["name", "value"],
                    "data": [["SECID", "RAGR"], ["ISIN", "DIFFERENT"]]}}
        with tempfile.TemporaryDirectory() as tmp, patch("recover_security_history.urlopen", return_value=BytesIO(json.dumps(payload).encode())):
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                collect_identity("RAGR", "RU000A0JQUZ6", Path(tmp))

    def _make_replay_fixture(self, directory):
        (directory / "iss_pages").mkdir()
        histories, old_rows = [], []
        def save(path, payload):
            data = json.dumps(payload).encode()
            path.write_bytes(data)
            return dict(file=path.name, sha256=hashlib.sha256(data).hexdigest())
        for event in VERIFIED_TRANSITIONS:
            identity = save(directory / f"{event.successor}_identity.json", {
                "description": {"columns": ["name", "value"], "data":
                    [["SECID", event.successor], ["ISIN", event.new_isin]]}})
            frame = pd.DataFrame([dict(TRADEDATE=event.first_tradable_date,
                ticker=event.successor, ISIN=event.new_isin, OPEN=100., CLOSE=101.)])
            csv = directory / f"{event.successor}.csv"
            frame.to_csv(csv, index=False)
            page = save(directory / "iss_pages" / f"{event.successor}_0.json",
                        {"history": {"columns": list(frame.columns), "data": frame.values.tolist()}})
            histories.append(dict(ticker=event.successor, rows=1, identity=identity, pages=[page],
                file=csv.name, sha256=hashlib.sha256(csv.read_bytes()).hexdigest()))
            old_rows.append(dict(TRADEDATE=pd.Timestamp(event.first_tradable_date),
                                 ticker=event.predecessor, OPEN=np.nan, CLOSE=np.nan))
        quotes = directory / "baseline.parquet"
        pd.DataFrame(old_rows).to_parquet(quotes, index=False)
        manifest = dict(transitions=transition_records(),
            inherited_same_ticker_custody_assumptions=[asdict(e) for e in SAME_TICKER_TRANSITIONS],
            through="2026-11-06", acknowledged=[e.predecessor for e in VERIFIED_TRANSITIONS],
            baseline_quote_sha256=hashlib.sha256(quotes.read_bytes()).hexdigest(), history=histories)
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return quotes

    def test_offline_replay_uses_verified_frozen_sources_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            quotes = self._make_replay_fixture(directory)
            with patch("recover_security_history.urlopen", side_effect=AssertionError("network forbidden")):
                result, audit = replay_verified_history(directory, quotes,
                    acknowledged_predecessors={"AGRO", "FIXP"}, through="2026-11-06")
            self.assertEqual(len(audit), 2)
            self.assertFalse(result.entry_allowed.any())
            self.assertAlmostEqual(result.loc[result.ticker.eq("AGRO"), "CLOSE"].iloc[0], 101*7.0135)

    def test_offline_replay_rejects_tampered_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            quotes = self._make_replay_fixture(directory)
            csv = directory / "RAGR.csv"
            csv.write_text(csv.read_text().replace("101.0", "999.0"))
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch: RAGR.csv"):
                replay_verified_history(directory, quotes,
                    acknowledged_predecessors={"AGRO", "FIXP"}, through="2026-11-06")


if __name__ == "__main__":
    unittest.main()
