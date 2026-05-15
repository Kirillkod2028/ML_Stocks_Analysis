# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML pipeline predicting `target = max(ADJ_HIGH[t+1..t+5]) / ADJ_CLOSE[t] - 1` (max 5-day return) for 35 MOEX blue-chip stocks. Data comes from the free ISS MOEX REST API (no auth required).

**Python >= 3.11** required (uses `X | Y` union type syntax throughout).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Build datasets (fetches from ISS MOEX, ~18 sec async)
python build_datasets_new.py   # preferred: includes split detection + target clipping
python build_datasets.py       # original version without split detection

# Step 2: Train models (runs as plain Python; also works as Jupyter notebook)
python train_models.py

# Optional: convert train_models.py to a Jupyter notebook
pip install jupytext
jupytext --to notebook train_models.py
jupyter notebook train_models.ipynb
```

On Windows, the entry-point sets `asyncio.WindowsSelectorEventLoopPolicy()` automatically.

In Jupyter (event loop already running): `await build_all_datasets_async()` instead of `asyncio.run(...)`.

## Architecture

### Two-stage pipeline

**Stage 1 — Dataset building** (`build_datasets_new.py` / `build_datasets.py`):

1. **Async I/O** (asyncio + aiohttp, semaphore `CONCURRENCY=6`):
   - Benchmarks fetched in parallel: IMOEX index, USD/RUB rate, RVI volatility index
   - Per ticker: `history` + `dividends` fetched concurrently via `asyncio.gather`
   - All 35 tickers fetched concurrently (semaphore-limited)
   - ISS uses cursor-based pagination — pages within one ticker are sequential

2. **Sync CPU processing** (pandas/numpy per ticker):
   - `_detect_and_fix_splits()` — detects daily price drops > 40% and rescales pre-split OHLC and dividends to post-split scale (new in `build_datasets_new.py`)
   - `backward_adjust()` — multiplicative backward dividend adjustment (Bloomberg/Refinitiv standard); includes anomaly guard that stops if ADJ_CLOSE drops below 1% of mean raw CLOSE
   - `calculate_features()` — computes 46 features across six groups: Momentum, Mean Reversion, Volatility, Volume/Liquidity, Technical, Cross-sectional/Beta
   - `add_calendar_features()` — day-of-week, month, days-to-ex-dividend, post-ex-dividend flag
   - `calculate_target()` — masks ±7 days around ex-dividend dates; clips to [-50%, +100%] (new in `build_datasets_new.py`)

3. **Normalization and output**:
   - **Market dataset**: cross-sectional winsorize (p1–p99) + rank-normalize per date → `datasets/market/dataset_market.parquet`
   - **Sector datasets**: same cross-sectional normalization within sector → `datasets/sectors/dataset_{SECTOR}.parquet`
   - **Per-ticker datasets**: rolling time-series z-score (120-day window, min 60 periods) → `datasets/tickers/dataset_{TICKER}.parquet`

**Stage 2 — Model training** (`train_models.py`, jupytext percent format `# %%`):

- Loads all three dataset scopes from parquet
- IC analysis (Spearman per date, cross-sectionally for panels)
- Two CV strategies:
  - `PurgedKFold` — expanding train window, purge + embargo (5 days each) to prevent lookahead
  - `WalkForwardCV` — expanding window with embargo, n sequential folds
- Optuna Bayesian HPO (TPESampler) maximizing mean Spearman IC on PurgedKFold
- Models: Ridge, ElasticNet, HuberRegressor, LightGBM — all wrapped in `Pipeline(Imputer → StandardScaler → Model)`
- Final models saved as `models/{dataset}_{model}.pkl` — a dict `{"pipeline": ..., "features": [...], "hparams": {...}, "dataset": "..."}` (not bare pipelines)
- Results, feature IC, feature importance tables, and figures saved to `results/`

### Key constants (both dataset scripts share the same values)

| Constant | Value | Purpose |
|---|---|---|
| `FETCH_START` | 2021-01-01 | ISS history fetch start |
| `DATA_START` | 2022-08-01 | Feature/target output start |
| `CONCURRENCY` | 6 | Max simultaneous ISS requests |
| `NAN_THRESHOLD` | 0.20 | Max allowed NaN fraction per row |
| `SPLIT_GAP_THRESHOLD` | 0.40 | Daily drop > 40% → treated as stock split |
| `TARGET_CLIP_LO/HI` | -0.50 / 1.00 | Physical bounds on 5-day target |

### Tickers and sectors

35 tickers in `TICKERS_SECTORS` dict mapped to 8 sectors: MOEXOG, MOEXFN, MOEXMM, MOEXCN, MOEXTL, ENERGY, IT, BUILD, OTHER.

### Feature groups (46 total)

Categorical (not normalized): `TSMOM`, `MACD_slope_sign`, `EMA_cross_20_50`, `day_of_week`, `month`, `post_ex_flag_10d`. All others are in `NORM_COLS` for normalization.

### Loading saved models

```python
import joblib
loaded = joblib.load("models/market_LightGBM.pkl")
pipeline = loaded["pipeline"]
features = loaded["features"]
predictions = pipeline.predict(X[features])
```
