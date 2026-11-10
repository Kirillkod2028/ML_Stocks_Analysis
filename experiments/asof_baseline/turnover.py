"""Controlled retention/turnover diagnostics on frozen out-of-fold predictions.

Selection uses after-close scores; orders execute at the next observed session's
open. Retained positions keep their share quantities, avoiding daily weight
rebalancing. Price-only outcomes exclude dividends and do not establish alpha.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio import (quote_entry_permissions, same_ticker_transition_records,
                       track_same_ticker_conversion, same_ticker_usage_records)


@dataclass
class Holding:
    shares: float
    mark: float
    entry_date: pd.Timestamp


ASSUMPTIONS = [
    "fractional shares and idealized observed open fills; final close liquidation",
    "same-open sale proceeds fund buys; synchronized execution is an assumption",
    "retained share quantities stay fixed; only available cash funds new names",
    "selection before execution; unavailable opening quotes never trigger substitutes",
    "entry_allowed=false quotes may value or close holdings, never open new ones",
    "same-ticker successor marks require custody eligibility in same_ticker_conversion_uses",
    "fixed retrospective stock universe and frozen development predictions",
    "no dividends, taxes, liquidity/capacity constraints or interest on cash",
    "stale marks make drawdown and volatility provisional",
]


def _summary(d: pd.DataFrame, *, side_bps: float, unresolved: int,
             trades: pd.DataFrame, **config) -> dict:
    realized = unresolved == 0
    stale_days = int(d.stale_positions.gt(0).sum())
    std = float(d.price_return.std(ddof=1))
    return dict(**config, side_bps=float(side_bps), sessions=len(d),
                start=str(d.TRADEDATE.iloc[0].date()), end=str(d.TRADEDATE.iloc[-1].date()),
                status=("unresolved_positions" if unresolved else
                        "complete_with_stale_marks" if stale_days else "complete"),
                price_only=True, dividends_included=False,
                realized_price_return=float(d.nav.iloc[-1] - 1.) if realized else None,
                marked_price_return=float(d.nav.iloc[-1] - 1.),
                annualized_price_return=(float(d.nav.iloc[-1] ** (252. / max(len(d)-1, 1)) - 1.)
                                         if realized else None),
                max_drawdown=float((d.nav / d.nav.cummax().clip(lower=1.) - 1.).min()),
                annualized_mean_over_vol=(float(d.price_return.mean()/std*np.sqrt(252))
                                          if std > 1e-12 else None),
                mean_exposure=float(d.exposure.mean()),
                mean_daily_turnover=float(d.turnover.mean()),
                total_notional_per_initial_nav=float(d.traded_notional.sum()),
                total_cost_per_initial_nav=float(d.costs.sum()),
                stale_nav_days=stale_days, unresolved_positions=int(unresolved),
                buys=int(trades.action.eq("BUY").sum()),
                sells=int(trades.action.isin(["SELL", "FINAL_SELL"]).sum()),
                missing_open_buys=int(trades.action.eq("SKIP_BUY_OPEN").sum()),
                ineligible_buys=int(trades.action.eq("SKIP_BUY_INELIGIBLE").sum()),
                gated_buys=int(trades.action.eq("SKIP_BUY_GATE").sum()),
                deferred_sell_attempts=int(trades.action.eq("DEFER_SELL").sum()),
                assumptions=ASSUMPTIONS)


def simulate_retention(predictions: pd.DataFrame, quotes: pd.DataFrame,
                       calendar: pd.DatetimeIndex, *, retain_rank: int = 10,
                       rebalance_every: int = 1, phase: int = 0,
                       side_bps: float = 0., end_date="2026-05-06",
                       max_positions: int = 5, enter_rank: int = 5,
                       same_ticker_events=None, cash_allocation: str = "selected_names"):
    """Keep names inside the retention band, fill vacancies from current top5.

    On a scheduled decision date, missing scores for a held name produce a sell
    order. Unfilled sells retry at each subsequent open, unless a later scheduled
    signal explicitly retains that name again. Buy lists last for one opening
    only. An unavailable sell still occupies a position slot; insufficient slots
    truncate the preselected buy list by its original signal rank. Missing buy
    quotes leave cash without replacement. Optional boolean ``can_enter`` and
    ``can_hold`` columns are known at signal time. Entry gates are applied after
    rank and vacancy preselection; no lower-ranked substitute is introduced.
    A held name also needs can_hold=True to remain in the retention band.

    ``selected_names`` preserves the original equal cash split over selected
    names. ``vacancy_slots`` divides available cash by the number of vacancies
    planned at the decision, or actual opening capacity if larger. Gated or
    missing buys therefore leave their allocation in cash. Failed sales occupy
    slots and never increase another selected name's allocation.

    The end date is an explicit predeclared evaluation boundary. All remaining
    positions are liquidated at its close if a quote exists; unclosed positions
    invalidate realized returns. Nothing is liquidated merely because a dataset
    has reached its last prediction row.
    """
    if not 0 <= side_bps < 10000 or max_positions < 1 or enter_rank < max_positions:
        raise ValueError("Invalid fees or portfolio size")
    if retain_rank < enter_rank or rebalance_every < 1 or not 0 <= phase < rebalance_every:
        raise ValueError("Invalid retention rank or rebalance phase")
    if cash_allocation not in {"selected_names", "vacancy_slots"}:
        raise ValueError("Unknown cash_allocation")
    same_ticker_events = (same_ticker_transition_records() if same_ticker_events is None
                          else same_ticker_events)
    conversion_uses = set()
    p = predictions[["TRADEDATE", "ticker", "prediction"]].copy()
    for column in ("can_enter", "can_hold"):
        if column in predictions:
            if not predictions[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise ValueError(f"{column} must contain only explicit nonmissing booleans")
            p[column] = predictions[column].to_numpy()
        else:
            p[column] = True
    p.TRADEDATE = pd.to_datetime(p.TRADEDATE)
    if p.empty or p.duplicated(["TRADEDATE", "ticker"]).any():
        raise ValueError("Predictions must have unique nonempty date/ticker keys")
    if not np.isfinite(p.prediction).all():
        raise ValueError("Nonfinite prediction")
    cal = pd.DatetimeIndex(calendar).drop_duplicates().sort_values()
    end = pd.Timestamp(end_date)
    if end not in cal:
        raise ValueError("Predeclared liquidation date absent from execution calendar")
    cal = cal[cal <= end]
    if not p.TRADEDATE.isin(cal).all():
        raise ValueError("Prediction date absent from evaluation calendar")
    first = int(cal.get_loc(p.TRADEDATE.min()))
    if first == len(cal)-1:
        raise ValueError("No execution sessions after first signal")
    signals = {day: part.sort_values(["prediction", "ticker"], ascending=[False, True])
               for day, part in p.groupby("TRADEDATE")}
    quote_columns = ["TRADEDATE", "ticker", "OPEN", "CLOSE"]
    if "entry_allowed" in quotes:
        quote_columns.append("entry_allowed")
    q = quotes[quote_columns].copy()
    q.TRADEDATE = pd.to_datetime(q.TRADEDATE)
    if q.duplicated(["TRADEDATE", "ticker"]).any():
        raise ValueError("Duplicate quote keys")
    entry_permissions = quote_entry_permissions(q)
    for col in ("OPEN", "CLOSE"):
        if (q[col].notna() & (~np.isfinite(q[col]) | q[col].le(0))).any():
            raise ValueError("Invalid observed price")
    observed = {(row.TRADEDATE, row.ticker): (row.OPEN, row.CLOSE)
                for row in q.itertuples(index=False)}
    held: dict[str, Holding] = {}
    pending_sells: dict[str, pd.Timestamp] = {}
    pending_buys: tuple[pd.Timestamp, list[tuple[str, bool]], int] | None = None
    cash = prev_nav = 1.
    fee_rate = side_bps / 10000.
    daily, records = [], []

    for i in range(first, len(cal)):
        day = cal[i]
        day_fee = day_notional = 0.
        def record(ticker, action, decision, notional=0., fee=0., shares=0., price=np.nan):
            records.append(dict(TRADEDATE=day, decision_date=decision, ticker=ticker,
                                action=action, notional=float(notional), cost=float(fee),
                                shares=float(shares), price=float(price)))

        # Execute only previously issued orders, before using today's close.
        for ticker, decision in list(pending_sells.items()):
            opening = observed.get((day, ticker), (np.nan, np.nan))[0]
            if not np.isfinite(opening):
                record(ticker, "DEFER_SELL", decision)
                continue
            position = held.pop(ticker)
            track_same_ticker_conversion(conversion_uses, same_ticker_events,
                                        ticker, position.entry_date, day)
            gross = position.shares * opening
            fee = gross * fee_rate
            cash += gross - fee
            day_fee += fee
            day_notional += gross
            record(ticker, "SELL", decision, gross, fee, position.shares, opening)
            del pending_sells[ticker]

        if pending_buys is not None:
            decision, candidates, planned_slots = pending_buys
            # Failed sells still count toward max_positions. Candidate truncation
            # follows previous-close ranking, never the existence of buy quotes.
            capacity = max_positions - len(held)
            chosen = [(ticker, allowed) for ticker, allowed in candidates
                      if ticker not in held][:capacity]
            denominator = (max(planned_slots, capacity) if cash_allocation == "vacancy_slots"
                           else len(chosen))
            budget = cash / denominator if chosen and denominator else 0.
            for ticker, allowed in chosen:
                if not allowed:
                    record(ticker, "SKIP_BUY_GATE", decision)
                    continue
                if not entry_permissions.get((day, ticker), True):
                    record(ticker, "SKIP_BUY_INELIGIBLE", decision)
                    continue
                opening = observed.get((day, ticker), (np.nan, np.nan))[0]
                if not np.isfinite(opening):
                    record(ticker, "SKIP_BUY_OPEN", decision)
                    continue
                if budget <= 1e-12:
                    record(ticker, "SKIP_BUY_CASH", decision)
                    continue
                gross = budget / (1.+fee_rate)
                fee = gross * fee_rate
                shares = gross / opening
                cash -= gross + fee
                held[ticker] = Holding(shares, opening, day)
                day_fee += fee
                day_notional += gross
                record(ticker, "BUY", decision, gross, fee, shares, opening)
            pending_buys = None

        stale = 0
        for ticker, position in list(held.items()):
            close = observed.get((day, ticker), (np.nan, np.nan))[1]
            if np.isfinite(close):
                track_same_ticker_conversion(conversion_uses, same_ticker_events,
                                            ticker, position.entry_date, day)
                position.mark = close
            else:
                stale += 1
            if day == end and np.isfinite(close):
                gross = position.shares * close
                fee = gross * fee_rate
                cash += gross - fee
                day_fee += fee
                day_notional += gross
                record(ticker, "FINAL_SELL", end, gross, fee, position.shares, close)
                del held[ticker]
                pending_sells.pop(ticker, None)

        if cash < -1e-10 or len(held) > max_positions:
            raise AssertionError("Cash borrowing or excess holdings")
        invested = sum(position.shares*position.mark for position in held.values())
        nav = float(cash + invested)
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError("Nonpositive/nonfinite portfolio value")
        daily.append(dict(TRADEDATE=day, nav=nav, cash=float(cash),
                          price_return=nav/prev_nav-1., exposure=invested/nav,
                          holdings=len(held), turnover=day_notional/prev_nav,
                          traded_notional=day_notional, costs=day_fee,
                          stale_positions=stale))
        prev_nav = nav

        # Every phase is anchored to the same first decision/calendar index.
        if day != end and (i-first) % rebalance_every == phase and day in signals:
            ordered = signals[day].ticker.tolist()
            ranks = {ticker: rank for rank, ticker in enumerate(ordered, 1)}
            enter_allowed = dict(zip(signals[day].ticker, signals[day].can_enter))
            hold_allowed = dict(zip(signals[day].ticker, signals[day].can_hold))
            retained = {ticker for ticker in held if ranks.get(ticker, np.inf) <= retain_rank
                        and hold_allowed.get(ticker, False)}
            for ticker in held:
                if ticker in retained:
                    pending_sells.pop(ticker, None)
                else:
                    pending_sells.setdefault(ticker, day)
            candidates = [ticker for ticker in ordered[:enter_rank] if ticker not in held]
            planned_slots = max_positions - len(retained)
            pending_buys = (day, [(ticker, bool(enter_allowed[ticker]))
                                  for ticker in candidates[:planned_slots]], planned_slots)

    d = pd.DataFrame(daily)
    t = pd.DataFrame(records, columns=["TRADEDATE", "decision_date", "ticker", "action",
                                        "notional", "cost", "shares", "price"])
    summary = _summary(d, trades=t, side_bps=side_bps, unresolved=len(held),
                       retain_rank=retain_rank, enter_rank=enter_rank,
                       max_positions=max_positions, rebalance_every=rebalance_every,
                       phase=phase, phase_origin=str(cal[first].date()),
                       end_liquidation_predeclared=True, cash_allocation=cash_allocation,
                       can_enter_gate_used="can_enter" in predictions,
                       can_hold_gate_used="can_hold" in predictions)
    summary["unresolved_tickers"] = sorted(held)
    summary["explicit_entry_permissions"] = "entry_allowed" in q
    summary["same_ticker_conversion_cohorts"] = len(conversion_uses)
    summary["same_ticker_conversion_uses"] = same_ticker_usage_records(conversion_uses,
                                                                      same_ticker_events)
    return d, t, summary


def equal_phase_composite(phases: list[tuple[pd.DataFrame, pd.DataFrame, dict]]):
    """Equal initial capital over all five phases, without cross-phase netting."""
    if len(phases) != 5 or {s["phase"] for _, _, s in phases} != set(range(5)):
        raise ValueError("Composite requires all five distinct fixed phases")
    keys = ("side_bps", "retain_rank", "enter_rank", "max_positions", "rebalance_every",
            "cash_allocation", "can_enter_gate_used", "can_hold_gate_used")
    configs = {tuple(s[key] for key in keys) for _, _, s in phases}
    if len(configs) != 1 or phases[0][2]["rebalance_every"] != 5:
        raise ValueError("Composite phases must share fees, sizing and five-session rules")
    index = pd.DatetimeIndex(phases[0][0].TRADEDATE)
    if any(not pd.DatetimeIndex(d.TRADEDATE).equals(index) for d, _, _ in phases):
        raise ValueError("Phase calendars differ")
    d = pd.DataFrame({"TRADEDATE": index})
    for column in ("nav", "cash", "costs", "traded_notional"):
        d[column] = np.mean([part[column].to_numpy() for part, _, _ in phases], axis=0)
    previous = d.nav.shift(1, fill_value=1.)
    d["price_return"] = d.nav/previous-1.
    d["turnover"] = d.traded_notional/previous
    d["exposure"] = (d.nav-d.cash)/d.nav
    d["stale_positions"] = np.sum([part.stale_positions for part, _, _ in phases], axis=0)
    d["holdings"] = np.sum([part.holdings for part, _, _ in phases], axis=0)
    trades = pd.concat([t.assign(phase=s["phase"], notional=t.notional/5., cost=t.cost/5.,
                                shares=t.shares/5.) for _, t, s in phases], ignore_index=True)
    unresolved = sum(s["unresolved_positions"] for _, _, s in phases)
    s = _summary(d, side_bps=phases[0][2]["side_bps"], trades=trades,
                 unresolved=unresolved, retain_rank=phases[0][2]["retain_rank"],
                 enter_rank=phases[0][2]["enter_rank"],
                 max_positions=5*phases[0][2]["max_positions"], rebalance_every=5, phase="equal_composite",
                 phase_origin=phases[0][2]["phase_origin"], end_liquidation_predeclared=True,
                 cash_allocation=phases[0][2]["cash_allocation"],
                 can_enter_gate_used=phases[0][2]["can_enter_gate_used"],
                 can_hold_gate_used=phases[0][2]["can_hold_gate_used"])
    s["assumptions"] = ASSUMPTIONS + ["equal initial capital across all five phases; no cross-phase netting"]
    s["unresolved_tickers"] = sorted({t for _, _, summary in phases for t in summary["unresolved_tickers"]})
    s["explicit_entry_permissions"] = all(summary["explicit_entry_permissions"]
                                           for _, _, summary in phases)
    s["same_ticker_conversion_uses"] = [dict(event, phase=summary["phase"])
                                         for _, _, summary in phases
                                         for event in summary["same_ticker_conversion_uses"]]
    s["same_ticker_conversion_cohorts"] = sum(summary["same_ticker_conversion_cohorts"]
                                              for _, _, summary in phases)
    return d, trades, s


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=here/"target_development_predictions.parquet")
    parser.add_argument("--policy", default="target_rank")
    parser.add_argument("--quotes", type=Path, default=here/"quotes.parquet")
    parser.add_argument("--calendar", type=Path, default=here/"calendar.parquet")
    parser.add_argument("--end", default="2026-05-06")
    parser.add_argument("--output", type=Path, default=here/"turnover_results")
    args = parser.parse_args()
    predictions = pd.read_parquet(args.predictions)
    predictions = predictions.loc[predictions.policy.eq(args.policy)]
    quotes = pd.read_parquet(args.quotes)
    calendar = pd.DatetimeIndex(pd.read_parquet(args.calendar).TRADEDATE)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    variants = [("daily_top5", 5, 1, 0), ("daily_buffer_top10", 10, 1, 0)]
    variants += [(f"buffer_every5_phase{phase}", 10, 5, phase) for phase in range(5)]
    for bps in (0., 10., 25.):
        phases = []
        for variant, retain, every, phase in variants:
            d, t, s = simulate_retention(predictions, quotes, calendar, retain_rank=retain,
                                         rebalance_every=every, phase=phase, side_bps=bps,
                                         end_date=args.end)
            s.update(variant=variant, policy=args.policy)
            tag = f"{variant}_{int(bps)}bps"
            d.to_csv(args.output/f"{tag}_daily.csv", index=False)
            t.to_parquet(args.output/f"{tag}_trades.parquet", index=False)
            summaries.append(s)
            if every == 5:
                phases.append((d,t,s))
        d, t, s = equal_phase_composite(phases)
        s.update(variant="buffer_every5_equal_phase_composite", policy=args.policy)
        tag = f"{s['variant']}_{int(bps)}bps"
        d.to_csv(args.output/f"{tag}_daily.csv", index=False)
        t.to_parquet(args.output/f"{tag}_trades.parquet", index=False)
        summaries.append(s)
    flat = [{k:v for k,v in s.items() if k not in
             {"assumptions", "unresolved_tickers", "same_ticker_conversion_uses"}} for s in summaries]
    pd.DataFrame(flat).to_csv(args.output/"summary.csv", index=False)
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    manifest = dict(policy=args.policy, end_liquidation_predeclared=args.end,
                    rules=variants, side_bps=[0,10,25], no_model_refitting=True,
                    future_quote_availability_used_for_selection=False,
                    explicit_entry_permissions="entry_allowed" in quotes,
                    primary_comparison="daily_top5 versus daily_buffer_top10",
                    phase_comparison="all five phases and equal-capital composite; no best-phase selection",
                    old_sleeve_comparison="descriptive only: entry/exit timing and exposure differ",
                    input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (args.predictions,args.quotes,args.calendar)},
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    same_ticker_custody_rules=same_ticker_transition_records(),
                    assumptions=ASSUMPTIONS)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(flat)[["variant", "side_bps", "status", "realized_price_return",
                              "mean_daily_turnover", "mean_exposure", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
