"""Paired monthly/yearly diagnostics, with all phases retained in the report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(directory: Path):
    summary = pd.read_csv(directory / "summary.csv")
    paired, aggregate = [], []
    for bps, group in summary.groupby("side_bps"):
        def load(variant):
            d = pd.read_csv(directory / f"{variant}_{int(bps)}bps_daily.csv",
                            parse_dates=["TRADEDATE"]).set_index("TRADEDATE")
            if d.index.has_duplicates or not d.index.is_monotonic_increasing:
                raise ValueError("Unordered or duplicate NAV dates")
            np.testing.assert_allclose(d.price_return, d.nav / d.nav.shift(1, fill_value=1.) - 1,
                                       atol=1e-12)
            return d
        baseline = load("daily_top5")
        base_record = group.set_index("variant").loc["daily_top5"]
        for row in group.itertuples(index=False):
            d = load(row.variant)
            if not d.index.equals(baseline.index):
                raise ValueError("Policy calendars differ")
            for frequency, label in (("M", "month"), ("Y", "year")):
                periods = d.index.to_period(frequency)
                left = (1. + d.price_return).groupby(periods).prod() - 1.
                right = (1. + baseline.price_return).groupby(periods).prod() - 1.
                for period, value in left.items():
                    dates = d.index[periods == period]
                    paired.append({"variant": row.variant, "side_bps": bps,
                                   "period_type": label, "period": str(period),
                                   "first_date": str(dates.min().date()),
                                   "last_date": str(dates.max().date()),
                                   "marked_period_return": value,
                                   "baseline_marked_period_return": right.loc[period],
                                   "delta_return": value - right.loc[period],
                                   "stale_days": int(d.loc[dates].stale_positions.gt(0).sum()),
                                   "baseline_stale_days": int(baseline.loc[dates].stale_positions.gt(0).sum())})
            monthly = [p for p in paired if p["variant"] == row.variant
                       and p["side_bps"] == bps and p["period_type"] == "month"]
            aggregate.append({"variant": row.variant, "side_bps": bps,
                              "realized_price_return": row.realized_price_return,
                              "delta_realized_vs_daily_top5": row.realized_price_return - base_record.realized_price_return,
                              "mean_daily_turnover": row.mean_daily_turnover,
                              "turnover_reduction_vs_daily_top5": 1. - row.mean_daily_turnover / base_record.mean_daily_turnover,
                              "mean_exposure": row.mean_exposure,
                              "positive_marked_months_vs_daily_top5": sum(p["delta_return"] > 0 for p in monthly),
                              "months": len(monthly), "status": row.status,
                              "unresolved_positions": row.unresolved_positions})
    pd.DataFrame(paired).to_csv(directory / "paired_periods.csv", index=False)
    pd.DataFrame(aggregate).to_csv(directory / "paired_summary.csv", index=False)
    notes = {"baseline": "daily_top5 in the same next-open engine",
             "phase_policy": "all five phases and their equal-initial-capital NAV composite; no best-phase selection",
             "monthly_returns": "marked NAV changes, provisional when stale holdings exist",
             "year_2026": "partial evaluation year ending 2026-05-06",
             "inference": "development execution diagnostics only; no independent-test significance claim",
             "dividends": "excluded; no total-return or guaranteed lower-bound interpretation"}
    (directory / "paired_notes.json").write_text(json.dumps(notes, indent=2))
    return pd.DataFrame(aggregate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    for directory in parser.parse_args().directories:
        result = summarize(directory)
        selected = result[result.side_bps.eq(10) & result.variant.isin(
            ["daily_top5", "daily_buffer_top10", "buffer_every5_equal_phase_composite"])]
        print(str(directory))
        print(selected.to_string(index=False))
