"""Parse bounded public broker snapshots into an UNVERIFIED review queue.

This intentionally cannot produce a corporate_actions event or coverage CSV:
no ex-date, verified_at, or shared price basis is assigned. Historical calendar
rows may have been forecasts; cash amounts may predate a share conversion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path


HERE = Path(__file__).resolve().parent
MONTHS = dict(zip("января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(), range(1, 13)))
DATE = r"\d{1,2} (?:" + "|".join(MONTHS) + r") 20\d{2}"
ROW = re.compile(rf"({DATE})\s*\n\s*({DATE})\s*\n\s*[^\n]+\s*\n\s*([\d ,\.]+) ₽")


def parse_date(value):
    day, month, year = value.split()
    return date(int(year), MONTHS[month], int(day))


def parse_snapshot(snapshot, start=date(2021, 1, 1), asof=date(2026, 11, 10)):
    table = snapshot.get("table", "").replace("\u00a0", " ")
    matches = list(ROW.finditer(table))
    if snapshot["table_available"] and not matches:
        raise ValueError(f"Present table cannot be parsed: {snapshot['ticker']}")
    # Every two explicit full dates must belong to one row. Stop on schema drift.
    if len(re.findall(DATE, table)) != 2 * len(matches):
        raise ValueError(f"Unparsed dividend dates: {snapshot['ticker']}")
    rows = []
    for index, match in enumerate(matches):
        last_cum, record = map(parse_date, match.group(1, 2))
        cash = float(match.group(3).replace(" ", "").replace(",", "."))
        if record < start:
            continue
        status = "historical_unverified" if record <= asof else "future_unverified"
        rows.append({"ticker": snapshot["ticker"], "source_row": index + 1,
                     "last_cum_date_candidate": last_cum.isoformat(),
                     "record_date_candidate": record.isoformat(),
                     "native_cash_amount_candidate": cash, "currency_candidate": "RUB",
                     "status": status, "cash_share_basis": "unverified_native_units",
                     "source_url": snapshot["source_url"],
                     "observed_on": snapshot["retrieved_on"],
                     "source_snapshot_sha256": hashlib.sha256(table.encode()).hexdigest()})
    pages = re.search(r"1 из (\d+)", table)
    audit = {"ticker": snapshot["ticker"], "source_url": snapshot["source_url"],
             "table_available": snapshot["table_available"], "parsed_rows": len(matches),
             "historical_candidates": sum(r["status"] == "historical_unverified" for r in rows),
             "future_candidates": sum(r["status"] == "future_unverified" for r in rows),
             "reported_pages": int(pages.group(1)) if pages else None,
             "history_exhaustive_verified": False, "issuer_reconciled": False,
             "access_error": snapshot.get("error"),
             "earliest_displayed_record": min((parse_date(m.group(2)).isoformat() for m in matches), default=None)}
    return rows, audit


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=HERE / "dividend_recovery/bulk_probe/bcs_table_snapshots.json")
    parser.add_argument("--output", type=Path, default=HERE / "dividend_recovery/bulk_probe")
    parser.add_argument("--asof", type=date.fromisoformat, default=date(2026, 11, 10))
    args = parser.parse_args()
    snapshots = json.loads(args.snapshots.read_text())
    rows, audit = [], []
    for snapshot in snapshots:
        r, a = parse_snapshot(snapshot, asof=args.asof)
        rows.extend(r)
        audit.append(a)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "bcs_unverified_candidates.csv", rows)
    write_csv(args.output / "bcs_access_audit.csv", audit)
    summary = {"asof": str(args.asof), "tickers_attempted": len(audit),
               "tickers_with_table": sum(a["table_available"] for a in audit),
               "historical_candidates": sum(a["historical_candidates"] for a in audit),
               "future_candidates_excluded_from_history": sum(a["future_candidates"] for a in audit),
               "verified_additional_tickers": 0,
               "snapshot_sha256": hashlib.sha256(args.snapshots.read_bytes()).hexdigest(),
               "limitations": ["First displayed page only; pagination not recovered.",
                               "Issuer approval, actual history completeness and share-unit conversions unverified.",
                               "Missing page or empty result never establishes zero dividends.",
                               "No first ex-date or reinvestable payment date inferred."]}
    (args.output / "bcs_candidate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
