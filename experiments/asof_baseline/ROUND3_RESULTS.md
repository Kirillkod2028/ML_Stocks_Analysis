# Corporate history and turnover experiment — 2026-11-10

The rank-model forecasts are frozen. This round changes execution rules and
repairs data coverage; it does not improve the forecast IC or use a new test set.
The rules and cost scenarios were recorded in `turnover_plan.json` before runs.

## Controlled execution comparison

Evaluation: 2024-01-03 through the predeclared close of 2026-05-06. All decisions
use the same stored rank forecasts, ending in April. Signals are formed after
close; orders execute at the next open. At most five names are held. Retained
shares are not reweighted, and only available cash funds new names. Same-open
sale proceeds funding purchases is an explicitly idealized netting assumption.

| Rule | Gross price return | 10 bps per side | 25 bps per side | Mean daily traded notional / prior NAV, at 10 bps |
| --- | ---: | ---: | ---: | ---: |
| Daily top-5, exit outside top-5 | +8.76% | -25.47% | -57.62% | 63.58% |
| Daily entry top-5, retain through top-10 | +9.28% | -3.97% | -20.84% | 21.68% |
| Every five sessions, all five phases equally funded | +1.70% | -4.05% | -12.05% | 9.75% |

The primary paired comparison is the two daily rules in the same engine.
Retention reduces turnover by 65.90%; their mean end-of-day exposure is the
same, about 99.66%. Gross performance changes by only +0.51 percentage points,
while the 10 bps net difference is +21.49 points. The result mainly comes from
lower costs. It is not yet a profitable 10 bps price-only strategy.

The net difference is positive in 2024 (+11.69 points), 2025 (+9.47 points) and
partial 2026 (+2.83 points). It is positive in 21/28 month bins through April,
or 22/29 including the partial May liquidation interval. Month/year returns
are marked NAV changes and provisional during suspensions.

All five rebalance phases are retained. Their 10 bps outcomes range from
-9.41% to +3.05%; the equal-initial-capital composite is -4.05%. Its NAV is the
average of phase NAVs with no cross-phase netting. The best phase is not selected.
The old next-open/fifth-close sleeve strategy remains a descriptive reference:
its timing, exposure and holding rule differ, so comparing it directly would
not isolate turnover policy.

## Security transformations

`security_continuity.py` and `recover_security_history.py` provide a separate
explicit custody scenario for existing holdings, not automatic ticker aliases.

| Predecessor | Successor | Shares per old receipt | First tradable successor date |
| --- | --- | ---: | --- |
| AGRO | RAGR | 7.0135 | 2025-02-17 |
| FIXP | FIXR | 158 | 2025-08-20 |

The collector recovered 439 RAGR and 310 FIXR observations through 2026-11-06,
verified actual SECID/ISIN from ISS metadata, and preserved source pages/hashes.
Old receipts are valued using successor prices times the verified ratio;
the actual trading pauses remain missing. No successor-based model predictions
are introduced. Sources, eligibility and assumptions are in
[SECURITY_CONTINUITY.md](SECURITY_CONTINUITY.md).

AGRO requires successful distribution to an eligible custody chain. FIXP
requires voluntary participation and completed exchange. Fractional shares,
zero conversion fees and no separately valued residual old claim are assumed.
The prior equal-universe sleeve diagnostic's eight remaining AGRO/FIXP positions
can be closed in this scenario: unresolved positions fall from eight to zero.
The rank retention results are identical with original and scenario quotes;
these policies did not hold those names through the transformations. Their
turnover benefit must not be attributed to the corporate-history repair.

The retained TCSG → T alias segment adds 181 sessions from 2024-03-18 to
2024-11-27 with the same ISIN. Earlier TCSG foreign receipts are excluded; the
segment preserves raw units and has not yet changed the panel or model forecasts.

OZON kept its ticker while changing legal security in 2025. Phases 1 and 3 and
four universe cohorts cross this event and explicitly assume completed 1:1
Russian-custody conversion. The two primary daily strategies sold before the
pause, so their controlled comparison does not rely on this assumption. Results
record affected entry cohorts and source-backed eligibility. Bridged AGRO/FIXP
quotes forbid new entries under the old ticker while permitting liquidation.

## Recovered dividend coverage

The reviewed ledger contains **88 entitlements for 11 of 35 tickers**.
The authoritative per-ticker interval is in `verified_coverage.csv`:

| Ticker | Events | Reviewed interval |
| --- | ---: | --- |
| SBER | 5 | 2021-01-01–2026-11-06 |
| LKOH | 11 | 2021-01-01–2026-11-06 |
| MOEX | 5 | 2021-01-01–2026-11-06 |
| MTSS | 7 | 2021-01-01–2026-11-06 |
| GAZP | 2 | 2021-01-01–2026-11-06 |
| IRAO | 6 | 2021-01-01–2026-11-06 |
| ROSN | 11 | 2021-01-01–2026-11-06 |
| NVTK | 12 | 2021-01-01–2026-11-06 |
| BSPB | 10 | 2021-01-01–2026-11-06 |
| PHOR | 16 | 2021-01-01–2025-12-31 |
| HYDR | 3 | 2021-01-01–2025-12-31 |

The adapter supplies known total-return labels for **11,483 of 35,662**
complete-price panel rows, including 248 dividend-bearing windows. Development
coverage is 6,320 of 19,459 labels, with 127 dividend-bearing windows. Unknown
tickers and uncovered periods remain NaN. PHOR and HYDR in 2026 are explicitly
unknown. These are manually reviewed bounded histories, not an unattended or
certified complete-market feed.

Two MTSS broker record dates were corrected against issuer decisions; IRAO uses
full-precision amounts. Same-date LKOH and PHOR components are retained separately.
Amount/record provenance, sourced last-cum dates and calendar-checked ex-sessions
are in [dividend_recovery/README.md](dividend_recovery/README.md). Future coverage,
mismatched share units and duplicate panel keys are rejected. Review timestamps
cannot be used as historical announcement timestamps.

The current execution comparisons remain price-only. Accrued receivable labels
do not pretend dividends are spendable cash on ex-date, and this partial ledger
must not be used to declare a full-universe total-return portfolio.

The broker review queue preserves 203 historical candidates for 31 tickers plus
independent BCS extracts. These overlap with each other and the verified ledger;
their counts must not be added as unique payments. FIXP redirects to FIXR are
quarantined. TATN/CHMF lack accessible primary histories, and TATN has unresolved
date conflicts. Other unreviewed or partial intervals remain work items, with
provenance and source-access limits recorded rather than zero-filled.

## Reproduction

Starting with the saved baseline panel/quotes/calendar, rank predictions and
`continuity_data` snapshots from the result archive. The offline replay validates
source hashes before reconstructing the custody scenario:

```bash
python experiments/asof_baseline/apply_verified_dividends.py
python experiments/asof_baseline/recover_security_history.py --offline --acknowledge AGRO FIXP
python experiments/asof_baseline/turnover.py
python experiments/asof_baseline/turnover.py --quotes experiments/asof_baseline/continuity_data/quotes_conversion_scenario.parquet --output experiments/asof_baseline/turnover_conversion_results
python experiments/asof_baseline/summarize_turnover.py experiments/asof_baseline/turnover_results experiments/asof_baseline/turnover_conversion_results
python experiments/asof_baseline/portfolio.py --predictions experiments/asof_baseline/target_development_predictions.parquet --policy target_rank --quotes experiments/asof_baseline/continuity_data/quotes_conversion_scenario.parquet --output experiments/asof_baseline/cohort_conversion_results
python -m unittest discover -s experiments/asof_baseline -p 'test_*.py'
```

The focused tests cover chronological signals, cash/fee conservation,
liquidation-only successor prices, custody assumptions and dividend data contracts.
The final run count and environment are recorded in the result archive.

For a fresh source acquisition, omit `--offline`; add `--recover-aliases` to
collect the separate same-ISIN TCSG segment. Fresh downloads may revise history.

The custody acknowledgement in the example is a research scenario, not evidence
that a particular account received shares. Source snapshots and complete numeric
results accompany the report archive; code and reviewed ledger are in GitHub.

Remaining limits: retrospective 35-name universe, adaptive development research,
no untouched test, incomplete dividend coverage, fractional shares, hypothetical
OHLC fills/costs, no capacity/lot/tax model, and stale valuations during pauses.
The next data gate is reviewing the remaining 24 issuers and the uncovered
2026 intervals before any full-universe total-return result. Retention through top-10 is the current
execution candidate; thresholds are not tuned further on these outcomes.
