# Broad dividend candidate collection

These files are an unverified review queue, separate from the reviewed SBER
and LKOH ledger. They add **zero** certified issuer coverage intervals.

On 2026-11-10, public BCS page extraction returned dividend tables for 29 of
the remaining 33 project tickers. The initial displayed pages contain 121
historical candidate rows since 2021 and three future rows. The future rows
are tagged separately and excluded from the historical count.

The actual access failures were AGRO and FIXP (403), IRAO and NVTK (404).
Other tickers are accessible but still unreviewed. Missing/failed pages never
mean no dividends. A standalone BCS HTTP request returned a challenge page;
public indexed page extraction supplied the usable snapshots.

BCS exposes up to five rows on the initial page and pagination controls.
Consequently, this collection does not recover older pages or establish
complete histories. Even a page reaching 2021 still needs issuer
reconciliation, including explicitly checking years without distributions.

`bcs_table_snapshots.json` preserves only the observed table block, source
URL, retrieval date and access result. Reproduce the parsed review queue:

```bash
python experiments/asof_baseline/collect_dividend_candidates.py
python experiments/asof_baseline/test_dividend_candidates.py
```

Candidates intentionally lack `ex_date`, `verified_at` and a usable adjusted
share basis. The parser preserves multiple same-date distributions: CHMF has
191.51 and 38.30 RUB on record date 2024-06-18. GMKN, PLZL, TRNFP, VTBR and T
also need share-unit checks before native dividend amounts can be combined
with adjusted prices. Cross-source agreement alone is not proof of an
actual payout or an exhaustive interval.

Next practical step is full-history broker pages, then reconcile each issuer's
approved distributions and missing years, verify last-cum dates, convert share
units using the corporate-action ledger, and only then add coverage rows to
the reviewed ledger. A separate Finam probe explores that full-history route.
