# Reviewed dividend recovery: eleven issuers

Reviewed 2026-11-10. This directory recovers **88 gross dividend
entitlements for eleven of the project's 35 tickers**: SBER, LKOH, MOEX, MTSS,
GAZP, IRAO, ROSN, NVTK, BSPB, PHOR and HYDR. It does **not** establish a complete-market
total-return target or a dividend-reinvesting portfolio.

| Scope | Complete price labels | Dividend history known | Labels containing dividend |
| --- | ---: | ---: | ---: |
| Actual saved panel | 35,662 | 11,483 | 248 |
| Development decisions from 2024-01-01; exit before 2026-05-07 | 19,459 | 6,320 | 127 |

Coverage expanded from 2,166 to 11,483 complete-price rows. For the other 24
tickers, dividend and total-return values remain NaN. The existing
`complete_labels` filter is preserved; endpoint prices alone do not reinstate
an incomplete baseline row. Historical coverage is a documented manual
completeness judgment, not a certified live feed.

## Evidence and dates

`reviewed_events.json` is the source of truth. Each event has a primary issuer
history/final disclosure (or completed NSD corporate-action bulletin) for the
amount and record date, a sourced last-cum date and an explicit ex-session.
`source_audit.json` lists the evidence and access limitations. Focused factual
extracts, rather than copied source pages, are in `snapshots/`.

| Issuer | Events | Primary amount/history evidence |
| --- | ---: | --- |
| SBER | 5 | Official dividend history, including explicit zero in 2022 |
| LKOH | 11 | Official history; distinct 537 and 256 RUB distributions in December 2022 |
| MOEX | 5 | Official full history and each approved AGM decision; explicit zero in 2022 |
| MTSS | 7 | Official annual totals, approved AGM/EGM disclosures and issuer results |
| GAZP | 2 | Completed AGM disclosures and NSD bulletin, annual report and no-annual-dividend decisions |
| IRAO | 6 | Issuer 2025 annual report p304 and 2026 actual-paid disclosure, with full precision |
| ROSN | 11 | Official cash/history and record-date tables |
| NVTK | 12 | Official payment history, 2020 final disclosure and current H1 2026 dividend page |
| BSPB | 10 | Official ordinary-share history, corporate calendar and issuer report |
| PHOR | 16 | Official history including separate same-day components; coverage stops 2025 |
| HYDR | 3 | Official 2025 report plus actual event notices; coverage stops 2025 |

All intervals start 2021-01-01. Nine end 2026-11-06; **PHOR and HYDR end
2025-12-31**, with 2026 explicitly unknown. For GAZP and IRAO, primary
reports plus final disclosures are reconciled with current full broker
histories to bound the 2026 coverage; this is a multi-source review, not a
primary-feed certification. Broker tables alone cannot establish completeness.
TATN/CHMF remain unreviewed because primary histories were inaccessible and
TATN has unresolved broker-date disagreements.

Concrete corrections found while reviewing:

- Finam shows MTSS record dates **2021-10-11 and 2022-07-11**; final issuer
  decisions establish **2021-10-12 and 2022-07-12**. The former dates are the
  first ex-sessions. Sourced last-cum dates are October 8 and July 8.
- Finam rounds IRAO cash. The ledger uses the issuer's exact published
  per-share amounts, including 0.321425305784583 RUB for FY2025.
- GAZP's absence of an annual dividend for a reporting period must not erase
  its 51.03 RUB interim entitlement in October 2022.
- The LKOH 537 and 256 RUB events share record date 2022-12-21 and ex-session
  2022-12-20. Both are counted (793 RUB), despite the issuer table's payment-year label.

`ex_date` is the first ex-session in the saved main-session calendar. The
adapter validates it against the first session after the sourced last-cum
date. It never subtracts fixed weekdays from a record date. This preserves
holidays and the T+2/T+1 transition. SBER 2023 is ex May 10 after May 8
last-cum; NVTK 2022 is ex May 4 after April 29 last-cum.

Individual event verification and interval completeness are separate.
Coverage includes a manual reconciliation statement and expected event count;
a known event alone does not create an interval. Future recommendations are
excluded. Review timestamps describe provenance, not historical public
announcement times; this ledger must not be used as a contemporaneous feature.

SBER and ROSN primary pages were accessible via indexed extraction despite
direct timeouts/503. Finam and BCS direct downloads may be challenge/403 pages;
those responses are not accepted as data. A reliable unattended refresh is
not established. URLs, factual snapshots and explicit access notes preserve
what was actually verified.

## Reproduce and use

From the repository root:

```bash
python experiments/asof_baseline/apply_verified_dividends.py
python experiments/asof_baseline/test_dividend_recovery.py
```

The adapter writes `verified_events.csv`, `verified_coverage.csv`,
`coverage_audit.csv`, `recovery_summary.json` and
`partial_total_return_labels.parquet`. The labels retain `TRADEDATE` and
ticker for one-to-one joins; ledger/panel SHA-256 hashes are in the summary.
The 18 focused tests cover event identity, calendar disagreements, issuer
reconciliation, annual totals, the two MTSS record corrections, IRAO precision,
GAZP interim entitlement, missing events and unknown coverage.

Amounts and prices use `split_adjusted_2026-11-06` share units. None of these
eleven issuers has a split conversion in the reviewed interval; amounts are
per ordinary share, never per ADR/GDR. Labels accrue gross receivables when
`entry_date < ex_date <= exit_date`. Actual client payment dates, tax,
settlement and reinvestment remain outside scope. No full-universe rank
retraining or portfolio total-return claim follows from partial recovery.

The bulk queues remain unverified discovery data even when a subset was
reviewed separately. Do not promote their records en masse. `expansion_sources.json`
preserves the ROSN/NVTK review bundle and unresolved TATN/CHMF leads.

Additional source corrections: HYDR legal record dates are July10 in both2021
and2022, despite Finam displaying July9 andJuly8. PHOR's 2023-07-11 payment
has separate216 and48 RUB components, and2024-07-11 has294 and15 RUB.
PHOR's future December2026 recommendation is excluded; neither PHOR nor
HYDR receives assumed zero dividends for2026. FEES exact2021 cash is known
but2024–2026 primary completeness and identity review remain open, so it was
not promoted.

The adapter rejects future/null/reversed coverage dates, incompatible input
share bases and duplicate date/ticker join keys. Explicitly reviewed zero-event
intervals are supported; unknown intervals still never become zero.
