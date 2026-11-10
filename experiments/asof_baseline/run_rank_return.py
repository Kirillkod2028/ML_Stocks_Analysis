"""Execute frozen rank/absolute/hybrid decisions under the recorded protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from turnover import simulate_retention

HERE = Path(__file__).resolve().parent
OUT = HERE / "rank_return_results"
POLICIES = ("rank_always", "return_always", "return_cash", "rank_cash")


def make_decisions(rows, policy, side_bps):
    if policy not in POLICIES or not 0 <= side_bps < 10000:
        raise ValueError("Unknown policy or invalid fee")
    r = rows[["TRADEDATE", "ticker", "rank_score", "expected_return"]].copy()
    if not np.isfinite(r[["rank_score", "expected_return"]]).all().all():
        raise ValueError("Nonfinite forecast")
    c = side_bps / 10000.
    net = (1. + r.expected_return) * (1. - c) / (1. + c) - 1.
    r["prediction"] = r.rank_score if policy.startswith("rank") else r.expected_return
    r["can_enter"] = net.gt(0) if policy.endswith("cash") else True
    # The entry fee already paid is not charged a second time in the hold gate.
    # This is a fixed continuation heuristic, not an optimal stopping solution.
    r["can_hold"] = r.expected_return.gt(0) if policy.endswith("cash") else True
    return r[["TRADEDATE", "ticker", "prediction", "can_enter", "can_hold"]]


def run(custody=False):
    destination = OUT / "custody" if custody else OUT
    destination.mkdir(parents=True, exist_ok=True)
    plan_path = HERE / "rank_return_plan.json"
    plan = json.loads(plan_path.read_text())
    for relative, expected in plan["input_sha256"].items():
        path = HERE.parents[1] / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    quotes_path = HERE / ("continuity_data/quotes_conversion_scenario.parquet" if custody else "quotes.parquet")
    quotes = pd.read_parquet(quotes_path)
    cal = pd.DatetimeIndex(pd.read_parquet(HERE / "calendar.parquet").TRADEDATE)
    end = pd.Timestamp(plan["liquidation_date"])
    if predictions.TRADEDATE.max() >= pd.Timestamp(plan["last_label_end_exclusive"]):
        raise ValueError("Predictions enter inspected holdout")
    summaries, period_rows, comparisons = [], [], []
    daily_frames = {}
    for block, rows in predictions.groupby("model_block"):
        for bps in plan["side_bps"]:
            for policy in POLICIES:
                decisions = make_decisions(rows, policy, bps)
                d, t, summary = simulate_retention(
                    decisions, quotes, cal, retain_rank=10, side_bps=bps,
                    end_date=end, cash_allocation="vacancy_slots")
                summary.update(model_block=block, policy=policy,
                               fully_cash_sessions=int(d.exposure.lt(1e-10).sum()),
                               mean_cash_weight=float((1-d.exposure).mean()))
                summaries.append(summary)
                tag = f"{block}_{policy}_{bps}bps"
                d.to_csv(destination / f"{tag}_daily.csv", index=False)
                t.to_parquet(destination / f"{tag}_trades.parquet", index=False)
                daily_frames[(block, bps, policy)] = d.set_index("TRADEDATE")
                print(tag, summary["realized_price_return"], flush=True)

            baseline = daily_frames[(block, bps, "rank_always")]
            for policy in POLICIES:
                d = daily_frames[(block, bps, policy)]
                if not d.index.equals(baseline.index):
                    raise ValueError("Different policy evaluation calendars")
                for frequency, name in (("M", "month"), ("Y", "year")):
                    periods = d.index.to_period(frequency)
                    returns = (1+d.price_return).groupby(periods).prod()-1
                    reference = (1+baseline.price_return).groupby(periods).prod()-1
                    for period in returns.index:
                        mask = periods == period
                        period_rows.append(dict(model_block=block, side_bps=bps,
                            policy=policy, period_type=name, period=str(period),
                            price_return=float(returns.loc[period]),
                            baseline_return=float(reference.loc[period]),
                            delta_return=float(returns.loc[period]-reference.loc[period]),
                            mean_exposure=float(d.loc[mask].exposure.mean()),
                            stale_days=int(d.loc[mask].stale_positions.gt(0).sum())))
                # Exclude the short final May liquidation interval from inference.
                delta = (d.price_return-baseline.price_return).loc[lambda s:s.index < "2026-05-01"]
                months = delta.groupby(delta.index.to_period("M")).mean()
                rng = np.random.default_rng(20261110)
                boot = months.to_numpy()[rng.integers(0, len(months), (10000,len(months)))].mean(axis=1)
                lo, hi = np.quantile(boot,[.025,.975])
                comparisons.append(dict(model_block=block, side_bps=bps, policy=policy,
                    mean_daily_return_delta=float(delta.mean()),
                    mean_month_daily_return_delta=float(months.mean()),
                    positive_months=int(months.gt(0).sum()), months=len(months),
                    exploratory_month_bootstrap_low=float(lo),
                    exploratory_month_bootstrap_high=float(hi)))

    flat = [{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in summaries]
    pd.DataFrame(flat).to_csv(destination / "execution_summary.csv", index=False)
    (destination / "execution_summary.json").write_text(json.dumps(summaries,indent=2,allow_nan=False))
    pd.DataFrame(period_rows).to_csv(destination / "paired_periods.csv", index=False)
    pd.DataFrame(comparisons).to_csv(destination / "paired_uncertainty.csv", index=False)
    reference = next(iter(daily_frames.values())).reset_index()[["TRADEDATE"]]
    reference["nav"] = 1.; reference["price_return"] = 0.; reference["exposure"] = 0.
    reference.to_csv(destination / "cash_zero_interest_daily.csv",index=False)
    manifest = dict(quote_scenario="custody" if custody else "original", quotes_sha256=hashlib.sha256(quotes_path.read_bytes()).hexdigest(), plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (Path(__file__),HERE/"turnover.py",HERE/"portfolio.py")},
        predictions_sha256=hashlib.sha256((OUT/"predictions.parquet").read_bytes()).hexdigest(),
        limits=plan["limits"], policies=list(POLICIES), fees=plan["side_bps"],
        cash_reference="NAV1, zero interest; not a money-market benchmark")
    (destination/"execution_manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custody", action="store_true")
    run(parser.parse_args().custody)
