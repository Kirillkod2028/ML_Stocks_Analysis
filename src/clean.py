"""
Reproducible cleaning pipeline for ML_Stocks_Analysis datasets.

Input:  datasets/{market,sectors,tickers}/*.parquet  (read-only)
Output: data/processed/{market,sectors,tickers}/*.parquet + metadata.json

Cleaning decisions are documented in reports/decisions.md.

Usage:
    python src/clean.py
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_ROOT = Path("datasets")
DST_ROOT = Path("data/processed")

# ---------------------------------------------------------------------------
# Constants (decisions.md)
# ---------------------------------------------------------------------------
DROP_COLS = ["Beta_USDRUB_60d"]
DTE_FILL_SENTINEL = 999

# Tickers confirmed to have no dividends in the observation window
NO_DIVIDEND_TICKERS = {"AGRO", "FEES", "FIXP", "OZON", "PIKK"}


# ---------------------------------------------------------------------------
# Core cleaning logic
# ---------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame, ticker_name: str | None = None) -> pd.DataFrame:
    """Apply all cleaning decisions to a single DataFrame.

    Args:
        df: Raw dataset loaded from parquet.
        ticker_name: Ticker symbol if this is a ticker-level dataset, else None.

    Returns:
        Cleaned DataFrame with the same column count (drop 1, add 1).
    """
    df = df.copy()

    # Decision 1: Drop zero-signal / all-NaN column
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Decision 2: Fill dte_ex_days NaN with sentinel 999
    if "dte_ex_days" in df.columns:
        df["dte_ex_days"] = df["dte_ex_days"].fillna(DTE_FILL_SENTINEL)

    # Decision 7: Add has_dividends binary flag
    # For ticker datasets: derive from ticker_name
    # For market/sector datasets: derive from the 'ticker' column if present
    if "has_dividends" not in df.columns:
        ticker_col = next((c for c in ["ticker", "TICKER"] if c in df.columns), None)
        if ticker_name is not None:
            df["has_dividends"] = np.int8(0 if ticker_name in NO_DIVIDEND_TICKERS else 1)
        elif ticker_col is not None:
            df["has_dividends"] = df[ticker_col].apply(
                lambda t: 0 if t in NO_DIVIDEND_TICKERS else 1
            ).astype(np.int8)
        else:
            df["has_dividends"] = np.int8(1)  # conservative default

    return df


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def _ticker_from_path(path: Path) -> str | None:
    """Extract ticker symbol from a ticker dataset filename like dataset_SBER.parquet."""
    stem = path.stem
    if path.parent.name == "tickers" and stem.startswith("dataset_"):
        return stem[len("dataset_"):]
    return None


def _build_metadata(df: pd.DataFrame, src: Path, dst: Path) -> dict:
    date_col = next((c for c in ["TRADEDATE", "DATE", "date"] if c in df.columns), None)
    date_start = str(pd.to_datetime(df[date_col]).min().date()) if date_col else None
    date_end = str(pd.to_datetime(df[date_col]).max().date()) if date_col else None

    nan_by_col = df.isnull().sum()
    nan_by_col = nan_by_col[nan_by_col > 0].to_dict()

    return {
        "source": str(src),
        "destination": str(dst),
        "rows": len(df),
        "cols": len(df.columns),
        "date_start": date_start,
        "date_end": date_end,
        "nan_remaining": {k: int(v) for k, v in nan_by_col.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    src_files = sorted(SRC_ROOT.rglob("*.parquet"))
    print(f"Found {len(src_files)} source files.")

    all_meta: dict = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "cleaning_rules": {
            "dropped_columns": DROP_COLS,
            "dte_ex_days_fill": DTE_FILL_SENTINEL,
            "no_dividend_tickers": sorted(NO_DIVIDEND_TICKERS),
            "added_columns": ["has_dividends"],
        },
        "datasets": {},
    }

    for src_path in src_files:
        rel = src_path.relative_to(SRC_ROOT)
        dst_path = DST_ROOT / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        ticker_name = _ticker_from_path(src_path)
        df_raw = pd.read_parquet(src_path)
        df_clean = clean_dataset(df_raw, ticker_name=ticker_name)

        df_clean.to_parquet(dst_path, index=False)

        key = str(rel)
        meta = _build_metadata(df_clean, src_path, dst_path)
        all_meta["datasets"][key] = meta

        nan_total = sum(meta["nan_remaining"].values())
        print(f"  {key}: {meta['rows']} rows, {meta['cols']} cols, {nan_total} NaN remaining")

    meta_path = DST_ROOT / "metadata.json"
    meta_path.write_text(json.dumps(all_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetadata saved: {meta_path}")
    print("Cleaning complete.")


if __name__ == "__main__":
    main()
