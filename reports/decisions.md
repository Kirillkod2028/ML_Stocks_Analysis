# Cleaning Decisions — ML_Stocks_Analysis

Generated: 2026-05-15

All cleaning is implemented in `src/clean.py`. Input: `datasets/`. Output: `data/processed/`.
Original files in `datasets/` are never modified.

---

## Decision 1: Drop `Beta_USDRUB_60d`

**Finding:** In market and sector datasets the column contains only the value 0.0 (std=0, unique=1).
In all 35 ticker datasets it is 100% NaN. In both cases it carries zero signal.

**Why it improves training:** A constant feature adds noise to regularised models (Ridge, ElasticNet)
and wastes a split in tree-based models (LightGBM). Removing it avoids spurious signal and
reduces dimensionality by 1.

**Implementation:** Drop column before saving every dataset scope.

---

## Decision 2: Fill `dte_ex_days` NaN with sentinel 999

**Finding:** Five companies never paid dividends in the observation window — AGRO, FEES, FIXP,
OZON, PIKK — and therefore have `dte_ex_days` = 100% NaN in their ticker datasets. Partial NaN
(7-88%) exists for other tickers on dates far from ex-dividend events.

**Why sentinel 999:** The feature encodes "days to next ex-dividend date". A very large positive
value (999) correctly signals "no upcoming ex-dividend event", which is a meaningfully different
regime from being 3 days before a dividend payment. This preserves the business semantics better
than mean/median imputation, which would confusingly place non-dividend stocks near the middle of
the dividend-cycle distribution.

**Scope:** Applied to all datasets where dte_ex_days is NaN (ticker, market, sector).

---

## Decision 3: Keep warm-up NaN rows (R12_1, R126, dist_52w_high)

**Finding:** Momentum indicators requiring 252 trading days of history (R12_1, R126, dist_52w_high)
are NaN for the first ~year of each ticker's history. This affects early rows for all tickers and
heavily affects T (77% NaN) and YDEX (54% NaN) due to their recent listing dates.

**Why keep:** These NaN values are structurally correct — the indicators genuinely cannot be
computed without enough history. Dropping rows would remove valid training signal from other
features. The Pipeline's SimpleImputer (median strategy, already defined in train_models.py)
handles these NaN at training time without leakage.

**Implementation:** No action — pass through as-is.

---

## Decision 4: Keep target outliers (target == 1.0)

**Finding:** Five rows in the market dataset have target = 1.0 (the clipping ceiling of +100%).
These come from extreme price action. The value is legitimate — it reflects an extreme 5-day rally
that was correctly clipped by the pipeline in build_datasets_new.py.

**Why keep:** The clipping already constrains the target to [-50%, +100%]. Further winsorisation
would introduce a second arbitrary threshold and information loss. HuberRegressor in the model
stack is already robust to these few extreme values.

---

## Decision 5: Keep short-history tickers T and YDEX

**Finding:** T (212 rows, from 2025-06-24) and YDEX (311 rows, from 2025-02-06) have very few
observations and high NaN rates in long-horizon momentum features.

**Why keep:** For market and sector datasets these tickers contribute valid training signal on
non-NaN features. Removing them would require separate dataset rebuild logic and is not warranted.
The Pipeline already uses SimpleImputer for remaining NaN.

**Metadata flag:** Both tickers are noted in `data/processed/metadata.json` with their actual
start dates and NaN percentages so downstream users can choose to exclude them in CV.

---

## Decision 6: Keep AGRO and FIXP with truncated date ranges

**Finding:** AGRO ends 2024-11-25 (likely delisted) and FIXP ends 2025-06-13.

**Why keep:** These tickers contribute valid historical training data up to their last trading
date. Excluding them would reduce training set size without benefit.

---

## Decision 7: Add `has_dividends` binary flag

**Rationale:** After filling `dte_ex_days` NaN with 999, the model needs a clean signal to
distinguish "far from ex-date" from "no ex-date exists". A binary `has_dividends` column (1 if the
ticker paid dividends in the dataset window, 0 otherwise) provides this explicitly.

**Scope:** All datasets. Per-ticker flag propagated into market and sector rows by ticker name.
Non-dividend tickers: AGRO, FEES, FIXP, OZON, PIKK (flag=0). All others: flag=1.

---

## Decision 8: Output format and folder structure

Cleaned files are saved to `data/processed/` mirroring the original structure:

```
data/processed/
  market/dataset_market.parquet
  sectors/dataset_{SECTOR}.parquet   (9 files)
  tickers/dataset_{TICKER}.parquet   (35 files)
  metadata.json                      (row counts, date ranges, NaN flags per file)
```

No train/val/test split is applied — splitting is handled by PurgedKFold / WalkForwardCV
inside `train_models.py` to avoid temporal leakage.

---

## Summary of Column Changes

| Column | Action | Scope |
|--------|--------|-------|
| `Beta_USDRUB_60d` | DROP | all 45 datasets |
| `dte_ex_days` NaN | Fill with 999 | all datasets where NaN exists |
| `has_dividends` | ADD (0/1 binary flag) | all 45 datasets |

Resulting columns per dataset: **50 -> 50** (drop 1, add 1).
