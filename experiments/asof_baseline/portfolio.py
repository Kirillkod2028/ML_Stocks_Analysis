"""Chronological price-only portfolio diagnostic with cash-funded sleeves.

Signals are selected using only decision-date predictions. Missing future prices
never remove a security from the selection set. Fractional shares and fills at
observed daily opens/closes are assumptions, not evidence of execution quality.
Dividends are deliberately excluded; this is not a total-return backtest.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ml_stocks_analysis.security_continuity import same_ticker_transition_records


@dataclass
class Position:
    ticker: str
    shares: float
    mark: float
    due_index: int
    decision_date: pd.Timestamp
    entry_date: pd.Timestamp


def track_same_ticker_conversion(used: set, events: list[dict], ticker: str,
                                entry_date: pd.Timestamp, quote_date: pd.Timestamp):
    """Record custody assumptions only when old holdings use successor quotes."""
    for i, event in enumerate(events):
        if (ticker == event["predecessor"]
                and entry_date <= pd.Timestamp(event["last_predecessor_session"])
                and quote_date >= pd.Timestamp(event["first_tradable_date"])):
            used.add((i, str(entry_date.date())))


def same_ticker_usage_records(used: set, events: list[dict]) -> list[dict]:
    return [dict(event, affected_entry_cohorts=len(dates), entry_dates=sorted(dates))
            for i, event in enumerate(events)
            if (dates := {entry for event_index, entry in used if event_index == i})]


def quote_entry_permissions(quotes: pd.DataFrame) -> dict:
    """Validate optional quote-level eligibility for opening new holdings.

    Synthetic continuity prices can value or liquidate an existing receipt but
    do not make that receipt available for purchase again. Absence of the field
    preserves ordinary quote behavior; an explicit column must be complete and
    strictly boolean rather than interpreting missing values or strings as true.
    """
    if "entry_allowed" not in quotes:
        return {}
    if not quotes.entry_allowed.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("entry_allowed must contain only nonmissing boolean values")
    return {(row.TRADEDATE, row.ticker): bool(row.entry_allowed)
            for row in quotes.itertuples(index=False)}


def quotes_from_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Recover observed prices, rejecting contradictory overlapping paths.

    A long suspension can extend past all eligible panel paths. Prefer the raw
    quote loader for production diagnostics; absent panel coverage stays absent.
    """
    parts = []
    dates = [pd.to_datetime(panel.TRADEDATE)]
    for h in range(1, 6):
        q = panel[[f"path_date{h}", "ticker", f"day{h}_open", f"path_close{h}"]].copy()
        q.columns = ["TRADEDATE", "ticker", "OPEN", "CLOSE"]
        q.TRADEDATE = pd.to_datetime(q.TRADEDATE)
        dates.append(q.TRADEDATE)
        parts.append(q.dropna(subset=["TRADEDATE"]))
    combined = pd.concat(parts, ignore_index=True)
    keys = ["TRADEDATE", "ticker"]
    if combined.groupby(keys)[["OPEN", "CLOSE"]].nunique().gt(1).any().any():
        raise ValueError("Conflicting quote values in overlapping panel paths")
    quotes = combined.groupby(keys, as_index=False)[["OPEN", "CLOSE"]].first()
    calendar = pd.DatetimeIndex(pd.concat(dates).dropna().unique()).sort_values()
    return quotes, calendar


def quotes_from_raw(raw: Path, tickers) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Load full stock paths with the same verified split units as the panel."""
    from build_panel import OHLC, read_prices, adjust_confirmed_splits

    market, _ = read_prices(raw / "IMOEX.csv")
    candidates = market.index[market[OHLC].notna().all(axis=1)]
    frames, observed = [], set()
    for ticker in sorted(tickers):
        frame, _ = read_prices(raw / f"{ticker}.csv")
        frame, events = adjust_confirmed_splits(frame, ticker)
        if any(e["status"] == "unexpected_ratio_NOT_adjusted" for e in events):
            raise ValueError(f"Unresolved corporate-action units for {ticker}")
        # A remaining extreme observed jump needs manual corporate-action review.
        if frame.CLOSE.dropna().pct_change(fill_method=None).abs().gt(.70).any():
            raise ValueError(f"Unexplained >70% price change for {ticker}")
        observed.update(frame.index[frame[OHLC].notna().all(axis=1)])
        frame = frame[["OPEN", "CLOSE"]].copy()
        frame["ticker"] = ticker
        frames.append(frame.reset_index())
    calendar = candidates[candidates.isin(observed)]
    return pd.concat(frames, ignore_index=True), pd.DatetimeIndex(calendar)


def simulate(predictions: pd.DataFrame, quotes: pd.DataFrame,
             calendar: pd.DatetimeIndex, *, side_bps: float = 0.,
             selection: str = "top5", holding_sessions: int = 5,
             end_date=None, same_ticker_events=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Use five independent sleeves; recycle only their available cash.

    Decision t -> entry open t+1 -> scheduled exit close t+5. A sleeve's next
    entry is t+6, so an exit's proceeds never fund an earlier same-day open.
    Missing entry opens leave cash, without replacing that stock. Missing exit
    closes postpone the exit until an observed close. At the next cycle, cash
    returned by other positions can fund the new cohort while suspended shares
    remain held. Same-day closing proceeds cannot fund morning entries.
    Unliquidated final positions invalidate realized results.
    """
    if not 0 <= side_bps < 10000 or holding_sessions < 1:
        raise ValueError("Invalid transaction cost or holding period")
    if selection not in {"top5", "universe"}:
        raise ValueError("selection must be top5 or universe")
    same_ticker_events = (same_ticker_transition_records() if same_ticker_events is None
                          else same_ticker_events)
    conversion_uses = set()
    p = predictions[["TRADEDATE", "ticker", "prediction"]].copy()
    p.TRADEDATE = pd.to_datetime(p.TRADEDATE)
    if p.empty or p.duplicated(["TRADEDATE", "ticker"]).any():
        raise ValueError("Predictions must have unique nonempty date/ticker keys")
    if not np.isfinite(p.prediction).all():
        raise ValueError("Nonfinite prediction")
    cal = pd.DatetimeIndex(calendar).drop_duplicates().sort_values()
    if end_date is not None:
        cal = cal[cal <= pd.Timestamp(end_date)]
    lookup = {d: i for i, d in enumerate(cal)}
    if not p.TRADEDATE.isin(cal).all():
        raise ValueError("Decision date absent from execution calendar")
    quote_columns = ["TRADEDATE", "ticker", "OPEN", "CLOSE"]
    if "entry_allowed" in quotes:
        quote_columns.append("entry_allowed")
    q = quotes[quote_columns].copy()
    q.TRADEDATE = pd.to_datetime(q.TRADEDATE)
    if q.duplicated(["TRADEDATE", "ticker"]).any():
        raise ValueError("Duplicate date/ticker quotes")
    entry_permissions = quote_entry_permissions(q)
    for name in ("OPEN", "CLOSE"):
        if ((q[name].notna()) & (~np.isfinite(q[name]) | q[name].le(0))).any():
            raise ValueError("Observed prices must be finite and positive")
    observations = {(r.TRADEDATE, r.ticker): (r.OPEN, r.CLOSE)
                    for r in q.itertuples(index=False)}
    scheduled = {}
    max_due = 0
    for day, rows in p.groupby("TRADEDATE", sort=True):
        i = lookup[day]
        due = i + holding_sessions
        if due >= len(cal):
            raise ValueError("A decision's planned exit extends beyond evaluation calendar")
        ordered = rows.sort_values(["prediction", "ticker"], ascending=[False, True])
        selected = ordered.head(5) if selection == "top5" else ordered
        scheduled[i + 1] = (i % holding_sessions, due, day, selected.ticker.tolist())
        max_due = max(max_due, due)
    cash = np.full(holding_sessions, 1. / holding_sessions)
    positions: list[list[Position]] = [[] for _ in range(holding_sessions)]
    cost = side_bps / 10000.
    daily, trades = [], []
    prev_nav = 1.
    cumulative_cost = 0.
    cumulative_traded = 0.
    skipped_open = skipped_cashless = delayed_exits = partial_recycles = 0
    skipped_ineligible = 0
    first = min(lookup[d] for d in p.TRADEDATE)
    for i in range(first, len(cal)):
        day = cal[i]
        day_cost = day_traded = 0.
        stale = 0
        if i in scheduled:
            sleeve, due, decision, selected = scheduled[i]
            if cash[sleeve] <= 1e-12:
                skipped_cashless += 1
                trades.append(dict(TRADEDATE=day, decision_date=decision, ticker="",
                                   sleeve=sleeve, action="SKIP_NO_CASH", notional=0., cost=0.))
            else:
                partial_recycles += int(bool(positions[sleeve]))
                allocation = float(cash[sleeve]) / len(selected)
                for ticker in selected:
                    if not entry_permissions.get((day, ticker), True):
                        skipped_ineligible += 1
                        trades.append(dict(TRADEDATE=day, decision_date=decision,
                                           ticker=ticker, sleeve=sleeve,
                                           action="SKIP_BUY_INELIGIBLE", notional=0., cost=0.))
                        continue
                    opening = observations.get((day, ticker), (np.nan, np.nan))[0]
                    if not np.isfinite(opening):
                        skipped_open += 1
                        trades.append(dict(TRADEDATE=day, decision_date=decision,
                                           ticker=ticker, sleeve=sleeve,
                                           action="SKIP_OPEN", notional=0., cost=0.))
                        continue
                    gross = allocation / (1. + cost)
                    fee = gross * cost
                    shares = gross / opening
                    cash[sleeve] -= gross + fee
                    positions[sleeve].append(Position(ticker, shares, opening, due, decision, day))
                    day_cost += fee
                    day_traded += gross
                    trades.append(dict(TRADEDATE=day, decision_date=decision,
                                       ticker=ticker, sleeve=sleeve, action="BUY",
                                       notional=gross, cost=fee))
        for sleeve, holdings in enumerate(positions):
            remaining = []
            for pos in holdings:
                closing = observations.get((day, pos.ticker), (np.nan, np.nan))[1]
                if np.isfinite(closing):
                    track_same_ticker_conversion(conversion_uses, same_ticker_events,
                                                pos.ticker, pos.entry_date, day)
                    pos.mark = closing
                else:
                    stale += 1
                if i >= pos.due_index and np.isfinite(closing):
                    gross = pos.shares * closing
                    fee = gross * cost
                    cash[sleeve] += gross - fee
                    day_cost += fee
                    day_traded += gross
                    delayed_exits += int(i > pos.due_index)
                    trades.append(dict(TRADEDATE=day, decision_date=pos.decision_date,
                                       ticker=pos.ticker, sleeve=sleeve, action="SELL",
                                       notional=gross, cost=fee))
                else:
                    remaining.append(pos)
            positions[sleeve] = remaining
        if cash.min() < -1e-10:
            raise AssertionError("Cash borrowing detected")
        invested = sum(pos.shares * pos.mark for sleeve in positions for pos in sleeve)
        nav = float(cash.sum() + invested)
        cumulative_cost += day_cost
        cumulative_traded += day_traded
        daily.append(dict(TRADEDATE=day, nav=nav, cash=float(cash.sum()),
                          price_return=nav / prev_nav - 1., exposure=invested / nav,
                          turnover=day_traded / prev_nav, costs=day_cost,
                          stale_positions=stale))
        prev_nav = nav
        if i >= max_due and not any(positions):
            break
    d = pd.DataFrame(daily)
    t = pd.DataFrame(trades, columns=["TRADEDATE", "decision_date", "ticker",
                                      "sleeve", "action", "notional", "cost"])
    unresolved = sum(len(x) for x in positions)
    stale_days = int(d.stale_positions.gt(0).sum())
    realized = unresolved == 0
    volatility = float(d.price_return.std(ddof=1))
    returns = float(d.nav.iloc[-1] - 1.) if realized else None
    summary = dict(selection=selection, side_bps=side_bps, sessions=len(d),
                   start=str(d.TRADEDATE.iloc[0].date()), end=str(d.TRADEDATE.iloc[-1].date()),
                   status="unresolved_positions" if unresolved else
                          ("complete_with_stale_marks" if stale_days else "complete"),
                   price_only=True, dividends_included=False,
                   realized_price_return=returns,
                   marked_price_return=float(d.nav.iloc[-1] - 1.),
                   annualized_price_return=(float(d.nav.iloc[-1] ** (252. / max(len(d)-1, 1)) - 1.)
                                            if realized else None),
                   max_drawdown=float((d.nav / d.nav.cummax().clip(lower=1.) - 1.).min()),
                   annualized_mean_over_vol=(float(d.price_return.mean() / volatility * np.sqrt(252))
                                             if volatility > 1e-12 else None),
                   total_notional_per_initial_nav=cumulative_traded,
                   total_cost_per_initial_nav=cumulative_cost,
                   mean_daily_turnover=float(d.turnover.mean()),
                   mean_exposure=float(d.exposure.mean()),
                   skipped_open_orders=skipped_open, skipped_cashless_sleeves=skipped_cashless,
                   skipped_ineligible_orders=skipped_ineligible,
                   explicit_entry_permissions="entry_allowed" in q,
                   same_ticker_conversion_cohorts=len(conversion_uses),
                   same_ticker_conversion_uses=same_ticker_usage_records(conversion_uses,
                                                                          same_ticker_events),
                   partial_recycles=partial_recycles,
                   delayed_exits=delayed_exits, stale_nav_days=stale_days,
                   unresolved_positions=unresolved,
                   assumptions=["fractional shares; observed daily open/close fills",
                                "five initially equal independent sleeves; no rebalancing between sleeves",
                                "available sleeve cash is recycled while suspended positions remain held",
                                "entry_allowed=false quotes cannot open new holdings; exits remain possible",
                                "same-ticker successor marks require the custody eligibility recorded in same_ticker_conversion_uses",
                                "fixed surviving ticker universe; no point-in-time membership",
                                "no dividend cashflows, taxes, borrow, or capacity model",
                                "cost bps are hypothetical per-side sensitivities",
                                "stale marks make drawdown and volatility provisional"])
    return d, t, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--panel", type=Path, default=here / "panel.parquet")
    parser.add_argument("--quotes", type=Path, default=here / "quotes.parquet")
    parser.add_argument("--calendar", type=Path, default=here / "calendar.parquet")
    parser.add_argument("--predictions", type=Path, default=here / "asof_development_predictions.parquet")
    parser.add_argument("--policy", help="Evaluate one frozen prediction policy")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--end", default="2026-05-06")
    parser.add_argument("--output", type=Path, default=here / "portfolio_results")
    args = parser.parse_args()
    predictions = pd.read_parquet(args.predictions)
    if args.policy is not None:
        predictions = predictions.loc[predictions.policy.eq(args.policy)]
        if predictions.empty:
            raise ValueError(f"Unknown prediction policy: {args.policy}")
    if args.raw_dir:
        panel = pd.read_parquet(args.panel)
        quotes, calendar = quotes_from_raw(args.raw_dir, panel.ticker.unique())
        quote_source = "complete_raw_history"
    elif args.quotes.exists():
        quotes = pd.read_parquet(args.quotes)
        calendar = (pd.DatetimeIndex(pd.read_parquet(args.calendar).TRADEDATE)
                    if args.calendar.exists() else
                    pd.DatetimeIndex(quotes.TRADEDATE.unique()).sort_values())
        quote_source = "complete_quote_grid"
    else:
        panel = pd.read_parquet(args.panel)
        quotes, calendar = quotes_from_panel(panel)
        quote_source = "panel_paths_limited_after_suspensions"
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for policy, rows in predictions.groupby("policy"):
        for selection in ("top5", "universe"):
            for bps in (0., 10., 25.):
                d, t, summary = simulate(rows, quotes, calendar, side_bps=bps,
                                          selection=selection, end_date=args.end)
                tag = f"{policy}_{selection}_{int(bps)}bps"
                d.to_csv(args.output / f"{tag}_daily.csv", index=False)
                t.to_parquet(args.output / f"{tag}_trades.parquet", index=False)
                summary["policy"] = str(policy)
                summary["quote_source"] = quote_source
                summaries.append(summary)
    pd.DataFrame([{k: v for k, v in s.items() if k not in {"assumptions", "same_ticker_conversion_uses"}}
                  for s in summaries]).to_csv(
        args.output / "summary.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    print(pd.DataFrame(summaries)[["policy", "selection", "side_bps", "status",
                                   "realized_price_return", "max_drawdown", "stale_nav_days"]].to_string(index=False))


if __name__ == "__main__":
    main()
