"""Compare fixed, monthly expanding and monthly rolling Ridge on dev dates.

Every monthly fit sees labels whose outcomes ended strictly before the month.
The existing fixed-fit OOF predictions are reused as a paired reference.
"""
import numpy as np
import pandas as pd

from run_baseline import HERE, HOLD, FOLDS, daily_scores, estimator, tune


def refit_predictions(data: pd.DataFrame, features: list[str], policy: str):
    parts = []
    for lo, hi in zip(FOLDS[:-1], FOLDS[1:]):
        test = data.loc[(data.TRADEDATE >= lo) & (data.TRADEDATE < hi)].copy()
        for month, te in test.groupby(test.TRADEDATE.dt.to_period("M")):
            start = te.TRADEDATE.min()
            tr = data.loc[data.label_end < start]
            if policy == "rolling_2y":
                tr = tr.loc[tr.TRADEDATE >= start - pd.DateOffset(years=2)]
            if tr.TRADEDATE.nunique() < 180:
                raise ValueError(f"Not enough train history for {policy} {month}")
            alpha = tune(tr, features)
            model = estimator(alpha).fit(tr[features], tr.y_excess5)
            pred = te[["TRADEDATE", "ticker", "y_return5", "y_excess5"]].copy()
            pred["prediction"] = model.predict(te[features])
            pred["policy"] = policy
            pred["train_label_end"] = tr.label_end.max()
            pred["alpha"] = alpha
            assert pred.train_label_end.max() < te.TRADEDATE.min()
            parts.append(pred)
    return pd.concat(parts, ignore_index=True)


def run():
    panel = pd.read_parquet(HERE / "panel.parquet")
    features = __import__("json").loads((HERE / "feature_sets.json").read_text())["base"]
    dev = panel.loc[(panel.TRADEDATE < HOLD) & (panel.label_end < HOLD)
                    & panel.complete_labels].copy()
    previous = pd.read_parquet(HERE / "development_predictions.parquet")
    previous = previous.loc[(previous.universe == "base_full")
                            & (previous.features == "base")].copy()
    previous["policy"] = "fixed_6m"
    cols = ["TRADEDATE", "ticker", "y_return5", "y_excess5", "prediction", "policy"]
    frames = [previous[cols]]
    for policy in ("expanding_monthly", "rolling_2y"):
        frames.append(refit_predictions(dev, features, policy)[cols])
    all_preds = pd.concat(frames, ignore_index=True)
    counts = all_preds.groupby(["policy", "TRADEDATE"]).size().unstack(0)
    assert counts.notna().all().all() and counts.nunique(axis=1).eq(1).all()
    keysets = [set(map(tuple, p[["TRADEDATE", "ticker"]].itertuples(index=False, name=None)))
               for p in frames]
    assert all(keys == keysets[0] for keys in keysets[1:])
    assert all_preds.TRADEDATE.max() < HOLD
    all_preds.to_parquet(HERE / "refit_development_predictions.parquet", index=False)
    rows, daily = [], {}
    for policy, part in all_preds.groupby("policy"):
        d = daily_scores(part)
        daily[policy] = d
        rows.append({"policy": policy, "dates": len(d), "rows": len(part),
                     "ic": float(d.ic.mean()),
                     "top_minus_universe": float(d.top_minus_universe.mean()),
                     "universe_minus_bottom": float(d.universe_minus_bottom.mean()),
                     "positive_ic_months": int((d.ic.groupby(d.index.to_period("M")).mean() > 0).sum())})
    ref = daily["fixed_6m"]
    paired = []
    rng = np.random.default_rng(20261110)
    for row in rows:
        d = daily[row["policy"]]
        row["delta_ic_vs_fixed"] = float((d.ic - ref.ic).mean())
        row["delta_top_vs_fixed"] = float((d.top_minus_universe-ref.top_minus_universe).mean())
        for metric in ("ic", "top_minus_universe"):
            delta = d[metric] - ref[metric]
            by_month = delta.groupby(delta.index.to_period("M")).mean()
            picks = rng.integers(0, len(by_month), size=(10000, len(by_month)))
            bootstrap = by_month.to_numpy()[picks].mean(axis=1)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            paired.append({"policy": row["policy"], "metric": metric,
                           "months": len(by_month), "delta_mean_daily": delta.mean(),
                           "delta_mean_month": by_month.mean(),
                           "positive_months": int((by_month > 0).sum()),
                           "monthly_bootstrap_low": low,
                           "monthly_bootstrap_high": high})
    pd.DataFrame(rows).to_csv(HERE / "refit_development_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(HERE / "refit_paired_months.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == "__main__":
    run()
