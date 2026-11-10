"""Controlled target comparison on development data, with monthly 2y Ridge fits.

All targets share training rows with complete price labels and finite historical
volatility normalization. Test rows are chosen from current information only;
missing future outcomes never suppress predictions. Rank and risk targets are
ranking scores, not forecasts in return units. Metrics conditional on observed
labels do not constitute executable portfolio returns or untouched-test evidence.
"""
from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from run_baseline import ALPHAS, FOLDS, HERE, HOLD, daily_scores, estimator

POLICIES = ("target_excess", "target_rank", "target_risk")
RISK_CLIP = 5.0  # Fixed before evaluation; not chosen on development outcomes.


def training_target(rows: pd.DataFrame, policy: str) -> pd.Series:
    """Transform only outcomes supplied by a completed historical training set."""
    if policy == "target_excess":
        return rows.y_excess5.copy()
    if policy == "target_risk":
        return rows.y_risk5.clip(-RISK_CLIP, RISK_CLIP)
    if policy == "target_rank":
        groups = rows.groupby("TRADEDATE").y_excess5
        counts = groups.transform("count")
        target = (groups.rank(method="average") - 1) / (counts - 1) - .5
        return target.where(counts > 1, 0.)
    raise ValueError(f"Unknown target: {policy}")


def predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
            policy: str, alpha: float) -> pd.DataFrame:
    y = training_target(train, policy)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite training target")
    model = estimator(alpha).fit(train[features], y)
    result = test[["TRADEDATE", "ticker", "y_return5", "y_excess5"]].copy()
    result["prediction"] = model.predict(test[features])
    return result


def tune_target(train: pd.DataFrame, features: list[str], policy: str) -> float:
    dates = np.sort(train.TRADEDATE.unique())
    if len(dates) < 90:
        raise ValueError("Insufficient train history for inner validation")
    boundary = pd.Timestamp(dates[-63])
    fit = train.loc[train.label_end < boundary]
    validation = train.loc[train.TRADEDATE >= boundary]
    if fit.TRADEDATE.nunique() < 25 or validation.empty:
        raise ValueError("Empty chronological inner split")
    assert fit.label_end.max() < validation.TRADEDATE.min()
    scores = []
    for alpha in ALPHAS:
        values = daily_scores(predict(fit, validation, features, policy, alpha))
        scores.append((float(values.ic.mean()), alpha))
    if not all(np.isfinite(score) for score, _ in scores):
        raise ValueError("Nonfinite inner validation IC")
    return max(scores, key=lambda pair: pair[0])[1]


def validate_transforms():
    """Small mathematical invariants; actual time/sample guards run on every fit."""
    dates = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"])
    rows = pd.DataFrame({"TRADEDATE": dates, "y_excess5": [-2., 0., 4., 9.],
                         "y_risk5": [-9., 0., 4., 9.]})
    np.testing.assert_allclose(training_target(rows, "target_rank"), [-.5, 0., .5, 0.])
    np.testing.assert_allclose(training_target(rows, "target_risk"), [-5., 0., 4., 5.])
    earlier = rows.iloc[:3]
    np.testing.assert_allclose(training_target(earlier, "target_rank"),
                               training_target(rows, "target_rank").iloc[:3])


def run():
    validate_transforms()
    panel_path = HERE / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    features = json.loads((HERE / "feature_sets.json").read_text())["base"]
    # label_end is computed from the common calendar, independent of stock quotes.
    dev = panel.loc[(panel.TRADEDATE < HOLD) & (panel.label_end < HOLD)].copy()
    test = dev.loc[dev.TRADEDATE >= FOLDS[0]]
    complete = dev.complete_labels & np.isfinite(dev.y_excess5)
    shared = complete & np.isfinite(dev.y_risk5)
    train_pool = dev.loc[shared]
    forecasts, fit_records = [], []
    for month, te in test.groupby(test.TRADEDATE.dt.to_period("M")):
        start = te.TRADEDATE.min()
        cutoff = start - pd.DateOffset(years=2)
        tr = train_pool.loc[(train_pool.label_end < start)
                            & (train_pool.TRADEDATE >= cutoff)]
        if tr.TRADEDATE.nunique() < 180:
            raise ValueError(f"Insufficient two-year history for {month}")
        assert tr.label_end.max() < start
        for policy in POLICIES:
            alpha = tune_target(tr, features, policy)
            pred = predict(tr, te, features, policy, alpha)
            pred["policy"] = policy
            pred["alpha"] = alpha
            pred["train_label_end"] = tr.label_end.max()
            forecasts.append(pred)
            fit_records.append({"policy": policy, "month": str(month),
                                "alpha": alpha, "train_rows": len(tr),
                                "train_dates": tr.TRADEDATE.nunique(),
                                "train_start": str(tr.TRADEDATE.min().date()),
                                "train_label_end": str(tr.label_end.max().date()),
                                "test_start": str(start.date()), "test_rows": len(te)})
        print(f"TARGET FIT {month}: train={len(tr)}, test={len(te)}", flush=True)

    predictions = pd.concat(forecasts, ignore_index=True)
    expected = set(test[["TRADEDATE", "ticker"]].itertuples(index=False, name=None))
    assert not predictions.duplicated(["policy", "TRADEDATE", "ticker"]).any()
    assert np.isfinite(predictions.prediction).all()
    assert predictions.TRADEDATE.max() < HOLD
    summary, daily = [], {}
    for policy, part in predictions.groupby("policy"):
        assert set(part[["TRADEDATE", "ticker"]].itertuples(index=False, name=None)) == expected
        values = daily_scores(part)
        daily[policy] = values
        summary.append({"policy": policy, "rows": len(part), "dates": len(values),
                        "unlabeled_predictions": int(part.y_return5.isna().sum()),
                        "conditional_daily_ic": values.ic.mean(),
                        "conditional_top_minus_universe": values.top_minus_universe.mean(),
                        "conditional_universe_minus_bottom": values.universe_minus_bottom.mean()})

    paired, monthly = [], []
    reference = daily["target_excess"]
    rng = np.random.default_rng(20261110)
    for policy in POLICIES:
        for metric in ("ic", "top_minus_universe"):
            delta = daily[policy][metric] - reference[metric]
            months = delta.groupby(delta.index.to_period("M")).mean()
            samples = rng.integers(0, len(months), size=(10000, len(months)))
            resampled = months.to_numpy()[samples].mean(axis=1)
            low, high = np.quantile(resampled, [.025, .975])
            paired.append({"policy": policy, "metric": metric, "months": len(months),
                           "delta_mean_daily": delta.mean(), "delta_mean_month": months.mean(),
                           "positive_months": int(months.gt(0).sum()),
                           "monthly_bootstrap_low": low, "monthly_bootstrap_high": high})
            monthly.extend({"policy": policy, "metric": metric, "month": str(month),
                            "delta": float(value)} for month, value in months.items())

    predictions.to_parquet(HERE / "target_development_predictions.parquet", index=False)
    pd.DataFrame(fit_records).to_csv(HERE / "target_fit_provenance.csv", index=False)
    pd.DataFrame(summary).to_csv(HERE / "target_development_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(HERE / "target_paired_months.csv", index=False)
    pd.DataFrame(monthly).to_csv(HERE / "target_monthly_deltas.csv", index=False)
    manifest = {"panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                "features": features, "policies": list(POLICIES), "alphas": list(ALPHAS),
                "risk_clip": RISK_CLIP, "fit_window_years": 2, "inner_validation_dates": 63,
                "test_start": str(test.TRADEDATE.min().date()),
                "test_end": str(test.TRADEDATE.max().date()),
                "holdout_boundary": str(HOLD.date()),
                "complete_development_rows": int(complete.sum()),
                "shared_training_eligible_rows": int(shared.sum()),
                "rows_removed_for_missing_risk_normalization": int((complete & ~shared).sum()),
                "prediction_eligibility_uses_future_quote_availability": False,
                "rank_target": "average within-date rank mapped to [-0.5,0.5] using (rank-1)/(n-1)-0.5",
                "risk_target": "clip(y_excess5/(past_sigma20*sqrt(5)), -5, 5); predictions used directly as ranking scores",
                "limits": ["Development experiment; no untouched-test claim",
                           "Fixed retrospective stock universe; survivorship bias remains",
                           "Metrics conditional on observed labels; use portfolio engine for execution",
                           "Month bootstrap assumes resampled month blocks are exchangeable; 28 blocks only",
                           "Price-only outcomes; dividends and costs absent"]}
    (HERE / "target_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == "__main__":
    run()
