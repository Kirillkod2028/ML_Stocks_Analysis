"""Build price-only, point-in-time base/EMA rows on a shared exchange calendar.

This experiment intentionally uses price returns only: no dividend dates or
cashflows are guessed. Confirmed split factors are unit conversions; features
are scale invariant, and unexplained price jumps remain visible in quality.json.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ml_stocks_analysis.data_pipeline import calculate_features, TICKERS_SECTORS, CATEGORICAL_COLS, CONFIRMED_SPLITS
from ema_features import ema_features

SPLITS = CONFIRMED_SPLITS
OHLC = ["OPEN", "HIGH", "LOW", "CLOSE"]
DROP_FEATURES = {"Beta_USDRUB_60d", "dte_ex_days", "has_dividends", "post_ex_flag_10d"}


def read_prices(path: Path, calendar=None):
    df = pd.read_csv(path)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    duplicate_count = int(df.duplicated("TRADEDATE").sum())
    df = df.drop_duplicates("TRADEDATE", keep="last").sort_values("TRADEDATE")
    df = df.set_index("TRADEDATE")
    for col in OHLC + ["WAPRICE", "VALUE", "VOLUME", "NUMTRADES"]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if calendar is not None:
        df = df.reindex(calendar)
    bad = ~np.isfinite(df[OHLC]).all(axis=1) | (df[OHLC] <= 0).any(axis=1)
    bad |= (df["HIGH"] < df[["OPEN", "CLOSE", "LOW"]].max(axis=1))
    bad |= (df["LOW"] > df[["OPEN", "CLOSE", "HIGH"]].min(axis=1))
    bad_observed = bad & df[OHLC].notna().any(axis=1)
    df.loc[bad, OHLC + ["WAPRICE", "VALUE", "VOLUME", "NUMTRADES"]] = np.nan
    return df, {"duplicate_dates": duplicate_count,
                "invalid_observed_ohlc": int(bad_observed.sum()),
                "invalid_ohlc_dates": [str(x.date()) for x in df.index[bad_observed]],
                "missing_ohlc_rows": int(bad.sum())}


def adjust_confirmed_splits(df, ticker):
    df = df.copy()
    records = []
    for date, factor in SPLITS.get(ticker, []):
        boundary = pd.Timestamp(date)
        before = df.loc[df.index < boundary, "CLOSE"].dropna()
        after = df.loc[df.index >= boundary, "CLOSE"].dropna()
        rec = {"ticker": ticker, "date": date, "factor": factor}
        if before.empty or after.empty:
            rec["status"] = "not_in_available_history"
        else:
            ratio = float(after.iloc[0] / before.iloc[-1])
            rec.update({"observed_ratio": ratio, "pre_date": str(before.index[-1].date()),
                        "post_date": str(after.index[0].date())})
            if .5 < ratio / factor < 2:
                mask = df.index < boundary
                df.loc[mask, OHLC + ["WAPRICE"]] *= factor
                df.loc[mask, "VOLUME"] /= factor
                rec["status"] = "adjusted"
            elif .5 < ratio < 2:
                rec["status"] = "already_adjusted_or_no_visible_jump"
            else:
                raise ValueError(f"{ticker} {date}: unverified split price ratio {ratio}")
        records.append(rec)
    return df, records


def make_targets(prices, market):
    """Five common sessions, after-close decision, next-open entry."""
    out = pd.DataFrame(index=prices.index)
    out["label_end"] = pd.Series(prices.index, index=prices.index).shift(-5)
    out["entry_date"] = pd.Series(prices.index, index=prices.index).shift(-1)
    out["entry_open"] = prices["OPEN"].shift(-1)
    out["exit_close"] = prices["CLOSE"].shift(-5)
    future_high = pd.concat([prices["HIGH"].shift(-h) for h in range(1, 6)], axis=1)
    future_valid = pd.concat([prices[OHLC].notna().all(axis=1).shift(-h, fill_value=False)
                             for h in range(1, 6)], axis=1).all(axis=1)
    market_entry = market["OPEN"].shift(-1)
    market_exit = market["CLOSE"].shift(-5)
    out["market_entry_open"] = market_entry
    out["market_return5"] = market_exit / market_entry - 1
    out["y_max5"] = (future_high.max(axis=1) / prices["CLOSE"] - 1).where(future_valid)
    out["y_return5"] = (out["exit_close"] / out["entry_open"] - 1).where(future_valid)
    out["y_excess5"] = out["y_return5"] - out["market_return5"]
    out["sigma"] = np.log(prices["CLOSE"]).diff().rolling(20, min_periods=20).std(ddof=1)
    out["y_risk5"] = out["y_excess5"] / (out["sigma"].where(out["sigma"] > 1e-8) * np.sqrt(5))
    for h in range(1, 6):
        out[f"path_date{h}"] = pd.Series(prices.index, index=prices.index).shift(-h)
        out[f"path_close{h}"] = prices["CLOSE"].shift(-h)
        out[f"market_close{h}"] = market["CLOSE"].shift(-h)
        for name in ("OPEN", "CLOSE"):
            out[f"day{h}_{name.lower()}"] = prices[name].shift(-h)
    out["complete_labels"] = out[["y_return5", "y_excess5"]].notna().all(axis=1)
    out["complete_labels"] &= np.isfinite(out[["y_return5", "y_excess5"]]).all(axis=1)
    return out


def test_target_indexing():
    calendar = pd.bdate_range("2024-01-01", periods=40, name="TRADEDATE")
    # Uneven daily moves make nonzero historical volatility and detect off-by-one.
    c = 100 + np.arange(40) + np.sin(np.arange(40))
    frame = pd.DataFrame({"OPEN": c-.2, "CLOSE": c, "HIGH": c+1, "LOW": c-1}, index=calendar)
    target = make_targets(frame, frame)
    np.testing.assert_allclose(target.iloc[22].y_return5, c[27]/(c[23]-.2)-1)
    np.testing.assert_allclose(target.iloc[22].y_excess5, 0)
    assert target.iloc[22].label_end == calendar[27]
    assert not target.iloc[-5:].complete_labels.any()
    frame.iloc[25, frame.columns.get_loc("CLOSE")] = np.nan
    target = make_targets(frame, frame)
    assert not target.iloc[20:25].complete_labels.any()


def test_feature_causality_and_units():
    rng = np.random.default_rng(20261109)
    calendar = pd.bdate_range("2020-01-01", periods=800, name="TRADEDATE")
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, len(calendar))))
    volume = rng.integers(10000, 50000, len(calendar)).astype(float)
    frame = pd.DataFrame({"TRADEDATE": calendar, "ADJ_OPEN": close * .999,
        "ADJ_HIGH": close * 1.01, "ADJ_LOW": close * .99, "ADJ_CLOSE": close,
        "ADJ_WAPRICE": close, "VOLUME": volume, "VALUE": close * volume,
        "NUMTRADES": rng.integers(100, 500, len(calendar))})
    original = calculate_features(frame).drop(columns="TRADEDATE")
    truncated = calculate_features(frame.iloc[:600]).drop(columns="TRADEDATE")
    np.testing.assert_allclose(original.iloc[:600], truncated, equal_nan=True,
                               atol=1e-10, rtol=1e-10)
    scaled = frame.copy()
    scaled[["ADJ_OPEN", "ADJ_HIGH", "ADJ_LOW", "ADJ_CLOSE", "ADJ_WAPRICE"]] *= .01
    scaled["VOLUME"] /= .01
    transformed = calculate_features(scaled).drop(columns="TRADEDATE")
    # calculate_features normalizes MACD by close in the corrected source.
    np.testing.assert_allclose(original, transformed, equal_nan=True, atol=1e-8, rtol=1e-7)


def main():
    t0 = time.perf_counter()
    test_target_indexing()
    test_feature_causality_and_units()
    raw = Path(os.environ.get("BASELINE_RAW_DIR", HERE / "raw"))
    needed = [raw / f"{ticker}.csv" for ticker in [*TICKERS_SECTORS, "IMOEX", "RVI"]]
    missing = [str(p.name) for p in needed if not p.exists()]
    if missing:
        raise RuntimeError(f"Collection incomplete: {missing}")
    market, mq = read_prices(raw / "IMOEX.csv")
    market = market[market[OHLC].notna().all(axis=1)].copy()
    # An index can publish an unchanged value when no equity has traded.
    # Only sessions with actual stock quotes can be decision/exit sessions.
    observed_sessions = set()
    for ticker in TICKERS_SECTORS:
        raw_dates = pd.read_csv(raw / f"{ticker}.csv", usecols=["TRADEDATE", "CLOSE"],
                                parse_dates=["TRADEDATE"])
        observed_sessions.update(raw_dates.loc[raw_dates.CLOSE.gt(0), "TRADEDATE"])
    market = market.loc[market.index.isin(observed_sessions)].copy()
    calendar = pd.DatetimeIndex(market.index, name="TRADEDATE")
    market_returns = np.log(market.CLOSE).diff()
    rvi = pd.read_csv(raw / "RVI.csv", parse_dates=["TRADEDATE"])
    rvi = rvi.drop_duplicates("TRADEDATE", keep="last").set_index("TRADEDATE")["CLOSE"]
    rvi = pd.to_numeric(rvi, errors="coerce").reindex(calendar)
    frames, quote_frames, ticker_quality, split_records = [], [], {}, []
    addon_cols = {}
    base_cols = None
    for ticker, sector in TICKERS_SECTORS.items():
        p, tq = read_prices(raw / f"{ticker}.csv", calendar)
        if tq["invalid_observed_ohlc"]:
            # Do not let a corrupted future CLOSE erase an otherwise available
            # OPEN and silently change which orders a backtest can execute.
            raise ValueError(f"{ticker}: invalid observed OHLC requires source repair")
        p, adjustments = adjust_confirmed_splits(p, ticker)
        quote_frames.append(p[["OPEN", "CLOSE", "VOLUME", "VALUE"]].reset_index().assign(ticker=ticker))
        split_records.extend(adjustments)
        daily_return = p.CLOSE.dropna().pct_change(fill_method=None).reindex(calendar)
        large = daily_return.abs() > .70
        tq["unexplained_abs_daily_return_gt70pct"] = [
            {"date": str(date.date()), "return": float(daily_return.loc[date])}
            for date in p.index[large]]
        if large.any():
            raise ValueError(f"{ticker}: unverified >70% observed price jump across sessions")
        for name in OHLC + ["WAPRICE"]:
            p[f"ADJ_{name}"] = p[name]
        f = calculate_features(p.reset_index(), imoex_ret=market_returns, rvi=rvi).set_index("TRADEDATE")
        f = f.drop(columns=list(DROP_FEATURES), errors="ignore")
        f["day_of_week"] = f.index.dayofweek
        f["month"] = f.index.month
        f["raw_RV20"] = f["RV20"]
        f["raw_ATR_pct"] = f["ATR_pct"]
        if base_cols is None:
            base_cols = f.columns.tolist()
        ema = ema_features(p.CLOSE)
        addon_cols["ema"] = ema.columns.tolist()
        features = pd.concat([f, ema], axis=1).replace([np.inf, -np.inf], np.nan)
        # No SSA-specific 250 consecutive-session restriction on the baseline.
        eligible = features[base_cols].notna().sum(axis=1) >= int(np.ceil(.8 * len(base_cols)))
        eligible &= p[OHLC].notna().all(axis=1)
        eligible &= p.index >= pd.Timestamp("2022-08-01")
        # Future targets are generated only after eligibility is determined.
        joined = pd.concat([features, make_targets(p, market)], axis=1)
        joined["ticker"] = ticker
        joined["sector"] = sector
        tq["eligible_rows_current_information"] = int(eligible.sum())
        tq["ema_available_rows"] = int((eligible & features[ema.columns].notna().all(axis=1)).sum())
        tq["eligible_rows_missing_future_labels"] = int((eligible & ~joined.complete_labels).sum())
        selected = joined.loc[eligible]
        tq["eligible_first"] = str(selected.index.min().date()) if len(selected) else None
        tq["eligible_last"] = str(selected.index.max().date()) if len(selected) else None
        ticker_quality[ticker] = tq
        frames.append(selected.reset_index())
        print(f"PANEL {ticker}: eligible={len(selected)} labels={int(selected.complete_labels.sum())}", flush=True)
    panel = pd.concat(frames, ignore_index=True).sort_values(["TRADEDATE", "ticker"]).reset_index(drop=True)
    # Never allow availability of a future label to influence today's CS ranks.
    preserved = {"RVI_level", "raw_RV20", "raw_ATR_pct"} | set(CATEGORICAL_COLS)
    rank_cols = [col for col in base_cols if col not in preserved]
    ranks = panel.groupby("TRADEDATE")[rank_cols].rank(pct=True, na_option="keep")
    means = ranks.groupby(panel.TRADEDATE).transform("mean")
    stds = ranks.groupby(panel.TRADEDATE).transform("std")
    panel[rank_cols] = (ranks - means) / stds.replace(0, np.nan)
    # A truly constant observed cross-section has neutral rank; preserve NaNs.
    zero_std = stds.eq(0) & ranks.notna()
    panel[rank_cols] = panel[rank_cols].mask(zero_std, 0)
    feature_sets = {"base": base_cols}
    feature_sets.update({key: base_cols + values for key, values in addon_cols.items()})
    assert not panel.duplicated(["TRADEDATE", "ticker"]).any()
    assert panel.loc[panel.complete_labels, "label_end"].gt(panel.loc[panel.complete_labels, "TRADEDATE"]).all()
    # Algebraic check: benchmark subtraction preserves contemporaneous rankings.
    complete = panel.loc[panel.complete_labels]
    raw_ranks = complete.groupby("TRADEDATE").y_return5.rank()
    excess_ranks = complete.groupby("TRADEDATE").y_excess5.rank()
    np.testing.assert_allclose(raw_ranks, excess_ranks)
    quality = {
        "price_return_only": True, "dividends_included": False,
        "source_end": str(calendar.max().date()), "calendar_rows": len(calendar),
        "market_quality": mq, "splits": split_records, "tickers": ticker_quality,
        "panel_rows": len(panel), "complete_label_rows": int(panel.complete_labels.sum()),
        "eligible_date_count": int(panel.TRADEDATE.nunique()),
        "date_start": str(panel.TRADEDATE.min().date()), "date_end": str(panel.TRADEDATE.max().date()),
        "ranks_computed_before_future_label_filter": True,
        "ssa250_history_required_for_all_variants": False,
        "ema_rows_current_information": int(panel[addon_cols["ema"]].notna().all(axis=1).sum()),
        "tests": ["fixed-calendar target indices", "incomplete future windows excluded",
                  "daily return/excess rank equality", "unique date/ticker", "label ends after decisions",
                  "baseline feature prefix causality", "baseline price/share unit invariance"],
        "seconds": time.perf_counter() - t0,
    }
    panel.to_parquet(HERE / "panel.parquet", index=False)
    pd.concat(quote_frames, ignore_index=True).to_parquet(HERE / "quotes.parquet", index=False)
    pd.DataFrame({"TRADEDATE": calendar}).to_parquet(HERE / "calendar.parquet", index=False)
    (HERE / "feature_sets.json").write_text(json.dumps(feature_sets, indent=2))
    (HERE / "quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps({k:quality[k] for k in ["panel_rows", "complete_label_rows", "date_start", "date_end", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
