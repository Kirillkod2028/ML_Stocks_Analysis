"""Apply a manually reviewed, explicitly bounded dividend ledger to real labels.

This is a bounded multi-issuer recovery, not a complete-market dividend feed. Last-cum
dates have independent source URLs and explicit expected first ex-sessions;
they are never guessed by subtracting weekdays from the record date.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from ml_stocks_analysis.corporate_actions import EVENT_COLUMNS, total_return_labels

PRICE_BASIS = "split_adjusted_2026-11-06"


def validated_ledger(review: dict, calendar: pd.DatetimeIndex):
    events = pd.DataFrame(review["events"])
    if events.empty:
        events = pd.DataFrame(columns=[*EVENT_COLUMNS, "record_date", "last_cum_date", "last_cum_source_url"])
    coverage = pd.DataFrame(review["coverage"])
    if not calendar.is_monotonic_increasing or not calendar.is_unique:
        raise ValueError("Calendar must be sorted and unique")
    for c in ("record_date", "last_cum_date", "ex_date"):
        events[c] = pd.to_datetime(events[c], errors="raise")
    for c in ("start_date", "end_date"):
        coverage[c] = pd.to_datetime(coverage[c], errors="raise")
    review_date = pd.Timestamp(review["review_date"])
    if pd.isna(review_date) or review_date.tzinfo is not None or review_date != review_date.normalize():
        raise ValueError("Review date must be a non-null naive calendar date")
    for frame, columns in ((events, ("record_date", "last_cum_date", "ex_date")),
                           (coverage, ("start_date", "end_date"))):
        for c in columns:
            if frame[c].isna().any() or frame[c].dt.tz is not None or not frame[c].eq(frame[c].dt.normalize()).all():
                raise ValueError(f"{c} must contain non-null naive calendar dates")
    if (coverage.end_date < coverage.start_date).any():
        raise ValueError("Reviewed coverage has reversed interval")
    if (coverage.end_date > review_date).any():
        raise ValueError("Reviewed coverage cannot extend beyond review date")
    if coverage[["ticker", "source_url", "expected_event_count"]].isna().any().any():
        raise ValueError("Reviewed coverage has missing required values")
    counts = pd.to_numeric(coverage.expected_event_count, errors="raise")
    if not (np.isfinite(counts) & counts.ge(0) & counts.eq(counts.round())).all():
        raise ValueError("Reviewed coverage event counts must be non-negative integers")
    if not coverage.source_url.str.match(r"https://[^/]+", na=False).all():
        raise ValueError("Reviewed coverage evidence missing")
    if events.duplicated(["ticker", "event_id"]).any():
        raise ValueError("Duplicate reviewed event")
    if (events.ex_date > events.record_date).any():
        raise ValueError("Reviewed ex-date after record date")
    if (events.ex_date > review_date).any():
        raise ValueError("Future events are not realized outcomes")
    for row in events.itertuples(index=False):
        if row.last_cum_date not in calendar:
            raise ValueError(f"Last cum date absent from calendar: {row.event_id}")
        future = calendar[calendar > row.last_cum_date]
        if not len(future) or future[0] != row.ex_date:
            raise ValueError(f"Reviewed ex-session disagrees with calendar: {row.event_id}")
        if not str(row.last_cum_source_url).startswith("https://"):
            raise ValueError("Last-cum evidence missing")
    for row in coverage.itertuples(index=False):
        actual = events.loc[(events.ticker == row.ticker)
                            & events.ex_date.between(row.start_date, row.end_date)]
        if len(actual) != row.expected_event_count:
            raise ValueError(f"Reviewed coverage count disagrees: {row.ticker}")
    if not set(events.ticker) <= set(coverage.ticker):
        raise ValueError("Every recovered issuer needs an explicit reviewed coverage")
    return events, coverage


def apply_to_panel(panel: pd.DataFrame, events: pd.DataFrame, coverage: pd.DataFrame):
    if panel[["TRADEDATE", "ticker"]].isna().any().any():
        raise ValueError("Panel has null label-join keys")
    if panel.duplicated(["TRADEDATE", "ticker"]).any():
        raise ValueError("Panel has duplicate TRADEDATE/ticker keys")
    if "price_basis" in panel and not panel.price_basis.eq(PRICE_BASIS).all():
        raise ValueError("Panel price_basis differs from expected split-adjusted units")
    valid = (panel.entry_date.notna() & panel.label_end.notna()
             & np.isfinite(panel.entry_open) & panel.entry_open.gt(0)
             & np.isfinite(panel.exit_close) & panel.exit_close.gt(0))
    if "complete_labels" in panel:
        valid &= panel.complete_labels.fillna(False)
    # Preserve unknown prices and unsupported tickers without manufacturing labels.
    rows = panel.loc[valid, ["TRADEDATE", "ticker", "entry_date", "label_end",
                             "entry_open", "exit_close"]].copy()
    rows = rows.rename(columns={"label_end": "exit_date", "entry_open": "entry_price",
                                "exit_close": "exit_price"})
    rows["price_basis"] = PRICE_BASIS
    result = total_return_labels(rows, events, coverage)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=HERE / "panel.parquet")
    parser.add_argument("--calendar", type=Path, default=HERE / "calendar.parquet")
    parser.add_argument("--review", type=Path,
                        default=HERE / "dividend_recovery/reviewed_events.json")
    parser.add_argument("--output", type=Path, default=HERE / "dividend_recovery")
    args = parser.parse_args()
    review = json.loads(args.review.read_text())
    calendar = pd.DatetimeIndex(pd.read_parquet(args.calendar).TRADEDATE)
    events, coverage = validated_ledger(review, calendar)
    panel = pd.read_parquet(args.panel)
    tickers = sorted(panel.ticker.unique())
    labels = apply_to_panel(panel, events, coverage)
    args.output.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output / "verified_events.csv", index=False)
    coverage.to_csv(args.output / "verified_coverage.csv", index=False)
    labels.to_parquet(args.output / "partial_total_return_labels.parquet", index=False)
    ticker_rows = []
    for ticker in tickers:
        p = labels.loc[labels.ticker == ticker]
        known = p.loc[p.dividends_known]
        ticker_rows.append({"ticker": ticker, "price_label_rows": len(p),
                            "covered_label_rows": len(known),
                            "labels_with_dividend": int(known.cash_dividends.gt(0).sum()),
                            "reviewed_events": int(events.ticker.eq(ticker).sum()),
                            "status": "issuer_history_reviewed" if len(known) else "unknown"})
    pd.DataFrame(ticker_rows).to_csv(args.output / "coverage_audit.csv", index=False)
    known = labels.loc[labels.dividends_known]
    dev = labels.loc[(labels.TRADEDATE >= "2024-01-01") & (labels.exit_date < "2026-05-07")]
    summary = {
        "review_date": review["review_date"], "recovered_tickers": sorted(coverage.ticker.unique()),
        "project_tickers": len(tickers), "reviewed_events": len(events),
        "price_label_rows": len(labels), "covered_label_rows": len(known),
        "labels_with_dividend": int(known.cash_dividends.gt(0).sum()),
        "development_price_rows": len(dev),
        "development_covered_rows": int(dev.dividends_known.sum()),
        "development_labels_with_dividend": int((dev.dividends_known & dev.cash_dividends.gt(0)).sum()),
        "all_uncovered_total_returns_are_nan": bool(labels.loc[~labels.dividends_known, "total_return"].isna().all()),
        "full_market_total_return_available": False,
        "portfolio_reinvestable_cash_available": False,
        "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
        "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        "limitations": review["limitations"],
    }
    (args.output / "recovery_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
