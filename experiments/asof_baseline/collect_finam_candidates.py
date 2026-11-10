"""Replay focused Finam table snapshots into an explicitly unverified queue.

This module neither infers ex-dates nor certifies interval completeness.
Live HTTP fetches returned 403; snapshots were retrieved by public web-page
extraction on 2026-11-10. Preserve the original table strings for audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "dividend_recovery/bulk_probe"


def parse_row(raw: str) -> dict:
    fields = [s.strip() for s in raw.split("|")]
    if len(fields) != 4:
        raise ValueError(f"Expected four table fields: {raw!r}")
    last_cum, record = [datetime.strptime(s, "%d.%m.%Y").date().isoformat()
                        for s in fields[:2]]
    match = re.fullmatch(r"([\d,\.\s\u00a0]+)\s*([^\d\s,\.]+)", fields[2])
    if not match:
        raise ValueError(f"Unknown amount/currency: {fields[2]!r}")
    cash = float(re.sub(r"\s", "", match[1]).replace(",", "."))
    currency = {"₽": "RUB", "$": "USD", "€": "EUR"}.get(match[2], match[2])
    if cash <= 0 or last_cum > record:
        raise ValueError(f"Invalid table row: {raw!r}")
    return {"last_cum_date": last_cum, "record_date": record,
            "cash_amount": cash, "currency": currency}


def collect(snapshot: dict, start: str, end: str):
    candidates, quarantined, audit = [], [], []
    for probe in snapshot["probes"]:
        rows = [parse_row(raw) for raw in probe["rows"]]
        in_range = 0
        for ordinal, row in enumerate(rows, 1):
            if not start <= row["record_date"] <= end:
                continue
            in_range += 1
            item = dict(row, ticker=probe["ticker"], source_row=ordinal,
                        source_url=probe.get("resolved_url", probe["url"]),
                        requested_url=probe["url"], source_reference=probe["reference"],
                        observed_at=probe["retrieved_on"], price_basis="unknown",
                        completeness="unknown", ex_date=None,
                        status="candidate_unverified" if probe["identity_confirmed"]
                        else "quarantined_identity_mismatch")
            (candidates if probe["identity_confirmed"] else quarantined).append(item)
        dates = [r["record_date"] for r in rows]
        audit.append({"ticker": probe["ticker"], "source_url": probe["url"],
                      "fetch_status": probe["fetch_status"],
                      "identity_confirmed": probe["identity_confirmed"],
                      "visible_history_rows": len(rows),
                      "window_rows": in_range,
                      "oldest_visible_record_date": min(dates) if dates else None,
                      "newest_visible_record_date": max(dates) if dates else None,
                      "visible_history_reaches_window_start": bool(dates and min(dates) <= start),
                      "complete_interval_certified": False,
                      "ex_dates_verified": False,
                      "price_basis_verified": False})
    return pd.DataFrame(candidates), pd.DataFrame(quarantined), pd.DataFrame(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=DATA / "finam_snapshots.json")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-11-10")
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshots.read_text())
    candidates, quarantined, audit = collect(snapshot, args.start, args.end)
    args.output.mkdir(exist_ok=True, parents=True)
    candidates.to_csv(args.output / "finam_candidates.csv", index=False)
    quarantined.to_csv(args.output / "finam_quarantined.csv", index=False)
    audit.to_csv(args.output / "finam_access_audit.csv", index=False)
    summary = {"window_start": args.start, "window_end": args.end,
               "probed_project_tickers": len(audit),
               "pages_with_table": int(audit.visible_history_rows.gt(0).sum()),
               "candidate_tickers": candidates.ticker.nunique(),
               "unverified_candidate_rows": len(candidates),
               "quarantined_rows": len(quarantined),
               "quarantined_tickers": sorted(quarantined.ticker.unique()),
               "no_table_tickers": audit.loc[audit.visible_history_rows.eq(0), "ticker"].tolist(),
               "pages_reaching_window_start": int(audit.visible_history_reaches_window_start.sum()),
               "same_date_rows_preserved": int(candidates.duplicated(["ticker", "record_date"], keep=False).sum()),
               "new_verified_events": 0, "new_complete_coverage_intervals": 0,
               "source_sha256": hashlib.sha256(args.snapshots.read_bytes()).hexdigest(),
               "direct_http_probe": {"CHMF": 403, "ROSN": 403, "MTSS": 403, "GAZP": 403},
               "notes": ["Date/cash candidates only; broker table is not issuer proof.",
                         "No ex-date inference; share basis and currency need issuer reconciliation.",
                         "FIXP redirects to FIXR; never attach that history to FIXP automatically.",
                         "Missing page or no parsed rows do not imply zero dividends."]}
    (args.output / "finam_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
