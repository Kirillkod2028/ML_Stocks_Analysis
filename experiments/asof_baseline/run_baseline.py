"""Development-only, chronological Ridge benchmark on price returns.

Uses the same decision dates for base/EMA comparisons. Previously inspected
May--October 2026 outcomes are never used for fitting or model selection.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
HOLD = pd.Timestamp("2026-05-07")
FOLDS = [pd.Timestamp(s) for s in
         ("2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01",
          "2026-01-01", "2026-05-07")]
ALPHAS = (10., 1000., 10000.)


def daily_scores(rows: pd.DataFrame) -> pd.DataFrame:
    """Equal weight per decision date, Spearman across tradable stocks."""
    valid = rows.sort_values(["TRADEDATE", "prediction", "ticker"],
                             ascending=[True, False, True])
    rank_p = valid.groupby("TRADEDATE").prediction.rank()
    rank_y = valid.groupby("TRADEDATE").y_return5.rank()
    dates = valid.TRADEDATE
    x = rank_p - rank_p.groupby(dates).transform("mean")
    y = rank_y - rank_y.groupby(dates).transform("mean")
    xy = (x * y).groupby(dates).sum()
    xx = x.pow(2).groupby(dates).sum()
    yy = y.pow(2).groupby(dates).sum()
    ic = (xy / np.sqrt(xx * yy)).where(xx.gt(0), 0.)
    top = valid.groupby("TRADEDATE").head(5).groupby("TRADEDATE").y_return5.mean()
    bottom = valid.groupby("TRADEDATE").tail(5).groupby("TRADEDATE").y_return5.mean()
    market = valid.groupby("TRADEDATE").y_return5.mean()
    out = pd.DataFrame({"ic": ic, "top_minus_universe": top - market,
                        "universe_minus_bottom": market - bottom,
                        "n_stocks": valid.groupby("TRADEDATE").size()})
    return out[out.n_stocks >= 10]


def estimator(alpha: float):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=alpha))


def score_model(train: pd.DataFrame, test: pd.DataFrame,
                columns: list[str], alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = estimator(alpha)
    model.fit(train[columns], train.y_excess5)
    q = test[["TRADEDATE", "ticker", "y_return5", "y_excess5"]].copy()
    q["prediction"] = model.predict(test[columns])
    return q, daily_scores(q)


def tune(train: pd.DataFrame, columns: list[str]) -> float:
    dates = np.sort(train.TRADEDATE.unique())
    if len(dates) < 90:
        raise ValueError("Insufficient train history for 63-session inner validation")
    start = pd.Timestamp(dates[-63])
    fit = train.loc[train.label_end < start]
    val = train.loc[train.TRADEDATE >= start]
    if fit.TRADEDATE.nunique() < 25 or len(val) == 0:
        raise ValueError("Inner chronological split is empty")
    assert fit.label_end.max() < val.TRADEDATE.min()
    results = []
    for alpha in ALPHAS:
        _, d = score_model(fit, val, columns, alpha)
        results.append((float(d.ic.mean()), alpha))
    if not all(np.isfinite(v[0]) for v in results):
        raise ValueError("Nonfinite inner IC")
    return max(results, key=lambda r: r[0])[1]


def run():
    panel = pd.read_parquet(HERE / "panel.parquet")
    panel["TRADEDATE"] = pd.to_datetime(panel.TRADEDATE)
    panel["label_end"] = pd.to_datetime(panel.label_end)
    feature_sets = json.loads((HERE / "feature_sets.json").read_text())
    base, ema = feature_sets["base"], feature_sets["ema"]
    dev = panel.loc[(panel.TRADEDATE < HOLD) & (panel.label_end < HOLD)
                    & panel.complete_labels].copy()
    shared = dev.loc[dev[[c for c in ema if c not in base]].notna().all(axis=1)]
    result, predictions = [], []
    for universe, data, candidates in [
        ("base_full", dev, {"base": base}),
        ("shared_ema_available", shared, {"base": base, "ema": ema}),
    ]:
        for name, columns in candidates.items():
            parts = []
            for index, (lo, hi) in enumerate(zip(FOLDS[:-1], FOLDS[1:])):
                tr = data.loc[data.label_end < lo]
                te = data.loc[(data.TRADEDATE >= lo) & (data.TRADEDATE < hi)]
                if tr.TRADEDATE.nunique() < 180 or te.TRADEDATE.nunique() < 20:
                    raise ValueError(f"Insufficient history for planned fold {index}")
                alpha = tune(tr, columns)
                q, daily = score_model(tr, te, columns, alpha)
                if len(daily) != te.TRADEDATE.nunique():
                    raise ValueError(f"Unscored decision date in fold {index}")
                assert tr.label_end.max() < te.TRADEDATE.min()
                q["universe"] = universe
                q["features"] = name
                q["fold"] = index
                parts.append(q)
                result.append({"universe": universe, "features": name,
                               "fold": index, "alpha": alpha,
                               "test_dates": len(daily), "test_rows": len(q),
                               "ic": daily.ic.mean(),
                               "top_minus_universe": daily.top_minus_universe.mean(),
                               "universe_minus_bottom": daily.universe_minus_bottom.mean(),
                               "train_label_end": str(tr.label_end.max().date()),
                               "test_start": str(te.TRADEDATE.min().date())})
            predictions.extend(parts)
    p = pd.concat(predictions, ignore_index=True)
    assert not p.TRADEDATE.ge(HOLD).any()
    summary = pd.DataFrame(result)
    summary.to_csv(HERE / "development_folds.csv", index=False)
    p.to_parquet(HERE / "development_predictions.parquet", index=False)
    aggregate = []
    for (universe, name), part in p.groupby(["universe", "features"]):
        d = daily_scores(part)
        aggregate.append({"universe": universe, "features": name,
                          "rows": len(part), "dates": len(d),
                          "ic": d.ic.mean(),
                          "top_minus_universe": d.top_minus_universe.mean(),
                          "universe_minus_bottom": d.universe_minus_bottom.mean()})
    pd.DataFrame(aggregate).to_csv(HERE / "development_summary.csv", index=False)
    print(pd.DataFrame(aggregate).to_string(index=False))


if __name__ == "__main__":
    run()
