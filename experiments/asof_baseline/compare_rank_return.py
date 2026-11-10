"""Frozen development comparison of rank scores and absolute price-return forecasts.

The absolute model is trained on gross next-open/fifth-close returns. Constant
execution costs transform expected returns afterward; subtracting a constant
from a Ridge target would merely move its intercept. Outcomes and calibration
statistics are conditional on observed complete labels. Unlabeled eligible
stocks still receive forecasts and must remain in execution selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from compare_targets import training_target
from run_baseline import ALPHAS, FOLDS, HERE, HOLD, daily_scores, estimator

MARKET_FEATURES = ["market_R5", "market_R21", "market_sigma20"]


def market_features(raw: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Information available at each decision close, without forward filling."""
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("Calendar must be sorted and unique")
    prices = raw[["TRADEDATE", "CLOSE"]].copy()
    prices["TRADEDATE"] = pd.to_datetime(prices.TRADEDATE)
    if prices.TRADEDATE.isna().any() or prices.TRADEDATE.duplicated().any():
        raise ValueError("Market dates must be nonmissing and unique")
    close = pd.to_numeric(prices.set_index("TRADEDATE").CLOSE, errors="raise")
    if (close.notna() & (~np.isfinite(close) | close.le(0))).any():
        raise ValueError("Invalid observed market close")
    close = close.reindex(calendar)
    return pd.DataFrame({"market_R5": close.pct_change(5, fill_method=None),
                         "market_R21": close.pct_change(21, fill_method=None),
                         "market_sigma20": np.log(close).diff().rolling(20, min_periods=20).std(ddof=1)},
                        index=calendar).rename_axis("TRADEDATE")


def net_expected_return(gross, side_bps: float):
    """Return on wealth committed including entry fee, net of terminal sale fee."""
    if not np.isfinite(side_bps) or not 0 <= side_bps < 10000:
        raise ValueError("Invalid side costs")
    c = side_bps / 10000.
    return (1. + gross) * (1. - c) / (1. + c) - 1.


def inner_split(train: pd.DataFrame):
    dates = np.sort(train.TRADEDATE.unique())
    if len(dates) < 90:
        raise ValueError("Insufficient historical inner-validation dates")
    boundary = pd.Timestamp(dates[-63])
    fit = train.loc[train.label_end < boundary]
    val = train.loc[train.TRADEDATE >= boundary]
    if fit.TRADEDATE.nunique() < 25 or val.empty:
        raise ValueError("Empty chronological inner split")
    if fit.label_end.max() >= val.TRADEDATE.min():
        raise AssertionError("Overlapping inner labels")
    return fit, val


def tune_and_fit(train, test, columns, target):
    fit, val = inner_split(train)
    def outcomes(rows):
        return (training_target(rows, "target_rank") if target == "rank"
                else rows.y_return5)
    scores = []
    for alpha in ALPHAS:
        model = estimator(alpha).fit(fit[columns], outcomes(fit))
        prediction = model.predict(val[columns])
        if target == "rank":
            scored = val[["TRADEDATE", "ticker", "y_return5"]].assign(prediction=prediction)
            value = float(daily_scores(scored).ic.mean())
        else:
            error = pd.Series((prediction-val.y_return5.to_numpy())**2, index=val.index)
            value = -float(error.groupby(val.TRADEDATE).mean().mean())
        if not np.isfinite(value):
            raise ValueError("Nonfinite inner objective")
        scores.append({"alpha": alpha, "objective": value,
                       "objective_name": "daily_ic" if target == "rank" else "negative_daily_mse",
                       "fit_label_end": str(fit.label_end.max().date()),
                       "validation_start": str(val.TRADEDATE.min().date()),
                       "validation_end": str(val.TRADEDATE.max().date())})
    alpha = max(scores, key=lambda x: x["objective"])["alpha"]
    model = estimator(alpha).fit(train[columns], outcomes(train))
    return model.predict(test[columns]), alpha, scores


def calibration(rows: pd.DataFrame) -> dict:
    valid = rows.loc[np.isfinite(rows.y_return5)].copy()
    valid["error"] = valid.expected_return-valid.y_return5
    valid["squared_error"] = valid.error**2
    valid["mean_squared_error"] = (valid.historical_mean_return-valid.y_return5)**2
    daily = valid.groupby("TRADEDATE")[["expected_return", "historical_mean_return", "y_return5",
                                             "error", "squared_error", "mean_squared_error"]].mean()
    mse = float(daily.squared_error.mean())
    ref = float(daily.mean_squared_error.mean())
    return {"labeled_rows": len(valid), "dates": len(daily),
            "mean_predicted_return": float(daily.expected_return.mean()),
            "mean_actual_return": float(daily.y_return5.mean()),
            "mean_error": float(daily.error.mean()), "daily_equal_mse": mse,
            "daily_equal_rmse": float(np.sqrt(mse)), "historical_mean_mse": ref,
            "mse_skill_vs_historical_mean": 1.-mse/ref if ref>0 else None,
            "positive_prediction_fraction": float(valid.expected_return.gt(0).mean())}


def run(output: Path, market_path: Path, plan: Path):
    if not plan.is_file():
        raise ValueError("Record the experiment protocol before running")
    panel_path = HERE/"panel.parquet"
    calendar_path = HERE/"calendar.parquet"
    feature_path = HERE/"feature_sets.json"
    prior_path = HERE/"target_development_predictions.parquet"
    panel = pd.read_parquet(panel_path)
    calendar = pd.DatetimeIndex(pd.read_parquet(calendar_path).TRADEDATE)
    base = json.loads(feature_path.read_text())["base"]
    market = market_features(pd.read_csv(market_path), calendar)
    panel = panel.merge(market.reset_index(), on="TRADEDATE", how="left", validate="many_to_one")
    if panel.duplicated(["TRADEDATE", "ticker"]).any():
        raise ValueError("Duplicate panel keys")
    dev = panel.loc[(panel.TRADEDATE < HOLD)&(panel.label_end < HOLD)].copy()
    test = dev.loc[dev.TRADEDATE >= FOLDS[0]]
    shared = dev.complete_labels & np.isfinite(dev.y_excess5) & np.isfinite(dev.y_risk5)
    train_pool = dev.loc[shared]
    if not np.isfinite(train_pool.y_return5).all():
        raise ValueError("Incomplete absolute training outcomes")
    blocks = {"base":base, "market_augmented":base+MARKET_FEATURES}
    forecasts, records, inner = [], [], []
    for month, te in test.groupby(test.TRADEDATE.dt.to_period("M")):
        start = te.TRADEDATE.min()
        tr = train_pool.loc[(train_pool.label_end < start)&
                            (train_pool.TRADEDATE >= start-pd.DateOffset(years=2))]
        if tr.TRADEDATE.nunique() < 180 or tr.label_end.max() >= start:
            raise ValueError("Insufficient or overlapping outer history")
        for block, columns in blocks.items():
            q = te[["TRADEDATE", "ticker", "y_return5", "y_excess5"]].copy()
            for target, column in (("rank", "rank_score"), ("absolute", "expected_return")):
                values, alpha, scores = tune_and_fit(tr, te, columns, target)
                q[column] = values
                records.append(dict(model_block=block, target=target, month=str(month), alpha=alpha,
                                    train_rows=len(tr),train_dates=tr.TRADEDATE.nunique(),
                                    train_start=str(tr.TRADEDATE.min().date()),
                                    train_label_end=str(tr.label_end.max().date()),
                                    test_start=str(start.date()),test_rows=len(te)))
                inner.extend(dict(model_block=block,target=target,month=str(month),**s) for s in scores)
            q["historical_mean_return"] = float(tr.y_return5.mean())
            q["model_block"] = block
            q["train_label_end"] = tr.label_end.max()
            forecasts.append(q)
        print(f"RANK/RETURN FIT {month}: train={len(tr)}, test={len(te)}",flush=True)
    predictions = pd.concat(forecasts,ignore_index=True)
    if predictions.duplicated(["model_block","TRADEDATE","ticker"]).any():
        raise AssertionError("Duplicate forecasts")
    if not np.isfinite(predictions[["rank_score","expected_return","historical_mean_return"]]).all().all():
        raise AssertionError("Nonfinite forecasts")
    expected = set(test[["TRADEDATE","ticker"]].itertuples(index=False,name=None))
    for _, part in predictions.groupby("model_block"):
        if set(part[["TRADEDATE","ticker"]].itertuples(index=False,name=None))!=expected:
            raise AssertionError("Forecast sample changed")
    previous = pd.read_parquet(prior_path)
    previous = previous.loc[previous.policy.eq("target_rank"),["TRADEDATE","ticker","prediction"]]
    matched = predictions.loc[predictions.model_block.eq("base")].merge(previous,on=["TRADEDATE","ticker"],validate="one_to_one")
    if len(matched)!=len(previous) or len(matched)!=len(test):
        raise AssertionError("Existing rank forecast keys differ")
    np.testing.assert_allclose(matched.rank_score,matched.prediction,atol=1e-12,rtol=1e-10)
    output.mkdir(parents=True,exist_ok=True)
    predictions.to_parquet(output/"predictions.parquet",index=False)
    pd.DataFrame(records).to_csv(output/"fit_provenance.csv",index=False)
    pd.DataFrame(inner).to_csv(output/"inner_scores.csv",index=False)
    summaries, monthly, bins, daily_rows, gates = [],[],[],[],[]
    for block, part in predictions.groupby("model_block"):
        summaries.append(dict(model_block=block,**calibration(part)))
        for month, group in part.groupby(part.TRADEDATE.dt.to_period("M")):
            monthly.append(dict(model_block=block,month=str(month),**calibration(group)))
        valid = part.loc[np.isfinite(part.y_return5)].copy()
        # Diagnostic bins only: no fit or trade rule uses these retrospective cutoffs.
        valid["bin"] = pd.qcut(valid.expected_return,5,labels=False,duplicates="drop")
        for group_id, group in valid.groupby("bin"):
            bins.append(dict(model_block=block,bin=int(group_id),rows=len(group),
                             forecast_min=group.expected_return.min(),forecast_max=group.expected_return.max(),
                             mean_forecast=group.expected_return.mean(),mean_outcome=group.y_return5.mean()))
        for target,column in (("rank","rank_score"),("absolute","expected_return")):
            scores = daily_scores(part.assign(prediction=part[column])).reset_index()
            scores["model_block"], scores["target"] = block,target
            daily_rows.append(scores)
            candidates = part.sort_values(["TRADEDATE",column,"ticker"],
                                           ascending=[True,False,True]).groupby("TRADEDATE").head(5).copy()
            candidates["predicted_net"] = net_expected_return(candidates.expected_return,10.)
            candidates["actual_net"] = net_expected_return(candidates.y_return5,10.)
            candidates["can_enter"] = candidates.predicted_net.gt(0)
            for allowed, group in candidates.groupby("can_enter"):
                known=group.loc[np.isfinite(group.y_return5)]
                gates.append(dict(model_block=block,ordering=target,side_bps=10.,can_enter=bool(allowed),
                                  selected_rows=len(group),unlabeled_selected_rows=int(group.y_return5.isna().sum()),
                                  labeled_rows=len(known),dates=group.TRADEDATE.nunique(),
                                  mean_predicted_net=float(known.predicted_net.mean()),
                                  conditional_mean_actual_net=float(known.actual_net.mean()),
                                  conditional_positive_net_fraction=float(known.actual_net.gt(0).mean())))
    pd.DataFrame(summaries).to_csv(output/"calibration_summary.csv",index=False)
    pd.DataFrame(monthly).to_csv(output/"calibration_monthly.csv",index=False)
    pd.DataFrame(bins).to_csv(output/"calibration_bins.csv",index=False)
    pd.concat(daily_rows,ignore_index=True).to_csv(output/"daily_metrics.csv",index=False)
    pd.DataFrame(gates).to_csv(output/"gate_diagnostics.csv",index=False)
    manifest = dict(features=blocks,alphas=list(ALPHAS),fit_years=2,inner_dates=63,
                    prediction_rows_per_block=len(test),decision_dates=test.TRADEDATE.nunique(),
                    unlabeled_rows_per_block=int(test.y_return5.isna().sum()),
                    test_start=str(test.TRADEDATE.min().date()),test_end=str(test.TRADEDATE.max().date()),
                    holdout_boundary=str(HOLD.date()),base_rank_matches_previous=True,
                    maximum_base_rank_difference=float(np.max(np.abs(matched.rank_score-matched.prediction))),
                    future_outcomes_used_for_prediction_eligibility=False,
                    costs_transform="(1+gross_prediction)*(1-side_cost)/(1+side_cost)-1",
                    tuning={"rank":"mean daily IC","absolute":"negative mean of daily mean squared errors"},
                    historical_mean="pooled completed gross training outcomes; fixed within each monthly fit",
                    input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [panel_path,calendar_path,feature_path,prior_path,market_path,plan]},
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    limits=["Adaptive development research, not independent holdout confirmation",
                            "Absolute return and rank use different appropriate inner selection metrics",
                            "Base momentum features are cross-section normalized; market augmentation is a separate sensitivity",
                            "Price-only outcomes: incomplete dividends, no cash interest or tax",
                            "Training and diagnostic labels conditional on observed completed outcomes",
                            "Forecast horizon is fixed five sessions; retention execution can hold longer",
                            "Calibration bins are retrospective diagnostics, never trading thresholds"])
    (output/"forecast_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"rank_return_results")
    parser.add_argument("--market",type=Path,default=HERE.parent/"ssa_targets_20261109/raw/IMOEX.csv")
    parser.add_argument("--plan",type=Path,default=HERE/"rank_return_plan.json")
    args=parser.parse_args()
    run(args.output,args.market,args.plan)
