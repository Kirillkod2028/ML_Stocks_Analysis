"""Dividend-aware labels with explicit source coverage and share-unit contracts.

This module does not fetch a speculative dividend feed. A caller supplies a
reviewed event ledger and intervals for which absence of events is verified.
Cash amounts and both prices must already use the same split-adjusted share
unit, identified by ``price_basis``. The module never infers ex-dates from a
record date or treats missing source coverage as zero dividends.

Returns accrue gross dividend *receivables* on the ex-date. They are suitable
for outcome labels, not a cash-reinvestment portfolio: a portfolio must also
track actual payment dates, tax and settlement separately. ``verified_at``
records data provenance, not the historical public announcement timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EVENT_COLUMNS = (
    "ticker", "event_id", "ex_date", "cash_amount", "currency",
    "price_basis", "source_url", "verified_at",
)
COVERAGE_COLUMNS = (
    "ticker", "start_date", "end_date", "source_url", "verified_at",
)
LABEL_COLUMNS = (
    "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
    "price_basis",
)


def _required(frame: pd.DataFrame, columns: tuple[str, ...], name: str):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{name}: null required values")


def _dates(frame: pd.DataFrame, columns: tuple[str, ...], name: str):
    for column in columns:
        dates = pd.to_datetime(frame[column], errors="raise")
        if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
            raise ValueError(f"{name}.{column}: expected naive calendar dates")
        frame[column] = dates


def _provenance(frame: pd.DataFrame, name: str):
    if not frame.source_url.str.match(r"https?://[^/]+", na=False).all():
        raise ValueError(f"{name}: missing source URL")
    if pd.to_datetime(frame.verified_at, errors="raise", utc=True).isna().any():
        raise ValueError(f"{name}: invalid verification timestamp")


def _interval_is_covered(intervals, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Require contiguous calendar-day coverage including both boundaries."""
    cursor = start
    for lo, hi in intervals:
        if hi < cursor:
            continue
        if lo > cursor:
            return False
        if hi >= end:
            return True
        cursor = hi + pd.Timedelta(days=1)
    return False


def total_return_labels(
    labels: pd.DataFrame,
    events: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Add gross cash-entitlement labels; leave uncovered outcomes unknown.

    A coverage row is an explicit assertion that the event ledger is complete
    for that ticker over [start_date, end_date]. An empty ``events`` table is
    only evidence of zero dividends *within a supplied complete interval*.
    Coverage need not be complete for every label: uncovered labels receive
    ``dividends_known=False`` and NaN dividend/total-return values.

    Entitlement condition is entry_date < ex_date <= exit_date, with entry at
    the opening and exit at the closing of their respective trading sessions.
    Buying on ex-date does not earn the dividend; selling on ex-date does.
    """
    labels, events, coverage = labels.copy(), events.copy(), coverage.copy()
    _required(labels, LABEL_COLUMNS, "labels")
    _required(events, EVENT_COLUMNS, "events")
    _required(coverage, COVERAGE_COLUMNS, "coverage")
    _dates(labels, ("entry_date", "exit_date"), "labels")
    _dates(events, ("ex_date",), "events")
    _dates(coverage, ("start_date", "end_date"), "coverage")
    _provenance(events, "events")
    _provenance(coverage, "coverage")
    for name, frame in (("labels", labels), ("events", events), ("coverage", coverage)):
        if not frame.ticker.map(lambda s: isinstance(s, str) and bool(s.strip())).all():
            raise ValueError(f"{name}: empty ticker")
    for name, frame in (("labels", labels), ("events", events)):
        if not frame.price_basis.map(lambda s: isinstance(s, str) and bool(s.strip())).all():
            raise ValueError(f"{name}: empty price_basis")
    if (labels.exit_date < labels.entry_date).any():
        raise ValueError("labels: exit before entry")
    if (coverage.end_date < coverage.start_date).any():
        raise ValueError("coverage: reversed interval")
    if events.duplicated(["ticker", "event_id"]).any():
        raise ValueError("events: duplicate event id")
    if not events.event_id.map(lambda s: isinstance(s, str) and bool(s.strip())).all():
        raise ValueError("events: empty event_id")
    if not events.currency.eq("RUB").all():
        raise ValueError("events: only explicitly RUB amounts are supported")
    for column in ("entry_price", "exit_price"):
        labels[column] = pd.to_numeric(labels[column], errors="raise")
        if not (np.isfinite(labels[column]) & labels[column].gt(0)).all():
            raise ValueError(f"labels: invalid {column}")
    events["cash_amount"] = pd.to_numeric(events.cash_amount, errors="raise")
    if not (np.isfinite(events.cash_amount) & events.cash_amount.gt(0)).all():
        raise ValueError("events: cash_amount must be finite and positive")
    intervals = {
        ticker: sorted(zip(part.start_date, part.end_date))
        for ticker, part in coverage.groupby("ticker")
    }
    event_groups = {ticker: part for ticker, part in events.groupby("ticker")}
    amounts, known = [], []
    for row in labels.itertuples(index=False):
        complete = _interval_is_covered(intervals.get(row.ticker, []), row.entry_date, row.exit_date)
        known.append(complete)
        if not complete:
            amounts.append(np.nan)
            continue
        part = event_groups.get(row.ticker)
        if part is None:
            amounts.append(0.)
            continue
        held = part.loc[(part.ex_date > row.entry_date) & (part.ex_date <= row.exit_date)]
        if not held.price_basis.eq(row.price_basis).all():
            raise ValueError(f"{row.ticker}: dividend and quote share units disagree")
        amounts.append(float(held.cash_amount.sum()))
    labels["dividends_known"] = known
    labels["cash_dividends"] = amounts
    labels["price_return"] = labels.exit_price / labels.entry_price - 1.
    labels["total_return"] = (labels.exit_price + labels.cash_dividends) / labels.entry_price - 1.
    return labels
