# Rank, absolute-return and cash-gate experiment — 2026-11-10

**The absolute-return model did not beat the rank model. The hybrid is only an
unconfirmed candidate: its primary net outcome improved, but month consistency,
calibration and drawdown do not support promotion.**

## Primary controlled comparison

Frozen development decisions: 2024-01-03–2026-04-28 (590 dates, 19,499 forecasts,
including 40 without complete future labels). All portfolios share the execution
calendar and final liquidation on 2026-05-06. These are cumulative returns for
the whole period, not annual rates. Training is monthly rolling two years.

Both models receive the same 45 baseline features and completed training rows.
Rank hyperparameters use purged chronological inner daily IC; absolute-return
hyperparameters use daily-equal MSE. This compares full learning procedures,
not the target transform in isolation. Base rank forecasts reproduce the prior
saved rank scores exactly.

| Policy | Gross price return | 10 bps/side | 25 bps/side | Mean exposure at 10 bps | Mean daily turnover at 10 bps | Marked max drawdown at 10 bps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rank, always invested | 8.19% | -4.86% | -21.49% | 99.58% | 21.57% | -31.66% |
| Absolute return, always invested | -10.93% | -28.91% | -49.27% | 99.63% | 37.94% | -42.88% |
| Absolute return + cash gate | 1.46% | -17.00% | -37.82% | 72.01% | 28.97% | -40.00% |
| Rank + absolute cash gate | 17.65% | -1.67% | -21.95% | 67.35% | 19.71% | -35.49% |

The all-cash reference returns 0%, with no interest. It beats all four primary
10 bps price-only policies; this is not a money-market benchmark.

**Why the rank baseline changed from -3.97% to -4.86%:** every policy now uses
five cash slots. A rejected or unfilled slot stays cash rather than being
redistributed over the remaining buys. The legacy default execution mode still
reproduces its old NAV. The new comparisons use this common sizing rule.

At 10 bps the hybrid improves cumulative return by 3.19 percentage points while
mean exposure drops from 99.58% to 67.35%. It is fully cash on 128 of 595 NAV
sessions, versus two initialization/final-liquidation sessions for rank alone.
In partial 2026 its mean exposure is just 8.76%. That change in market exposure
is part of the outcome, not independent evidence of timing skill.

Hybrid minus rank has positive monthly returns in only 9 of 28 January 2024–April
2026 months. Exploratory month-block bootstrap of mean daily-return differences
has 95% interval approximately [-0.000448, +0.000515], including zero. It resamples
28 month means and is not a confidence interval for compounded total return or
an independent-test claim. The short May liquidation interval is excluded from
that inference. Marked drawdown worsens from -31.66% to -35.49%; stale quotes make
both risk estimates provisional. The cash rule has not established lower risk.

## What the two models predict

The rank target is the within-date rank of completed five-session returns. The
absolute model predicts gross price return from the next open to the fifth
close. Expected net return for per-side proportional cost c is:

    expected_net = (1 + expected_gross) * (1 - c) / (1 + c) - 1

Constant fees therefore transform the forecast; no claim is made that merely
subtracting a fixed fee creates a more informative target. Ridge includes an
intercept; see the [official model documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

The four policies use the same next-open engine and rank<=10 retention band:

- Rank always: order by rank score; no monetary threshold.
- Absolute always: order by expected return; no threshold. This is a control.
- Absolute + cash: same order, enter only if expected net>0.
- Rank + cash: rank order, same absolute-model threshold.

Cash variants also require expected gross>0 to retain existing holdings; sunk
entry fees are not charged again. This is a fixed rolling-signal heuristic, not
an optimal stopping rule. Five-session prediction horizons and actual holding
periods differ. Top-5 selection precedes entry filtering; rank6 never replaces a
rejected rank1. Signals use the close, fills use the next open. Missing sells
retain positions and cannot create borrowing. Thresholds are fixed, not tuned.

Changing fees changes both the cash gate and trades, so cost columns are complete
policy scenarios, not a pure subtraction of fees from one identical trade list.

## Calibration and sensitivity

Base rank daily IC is 0.09466 versus 0.05626 for absolute return. Across labeled
observations, the daily-equal mean absolute forecast is +0.2727% while the outcome
is -0.2498%. Its MSE is 1.56% worse than forecasting the historical training mean.
In rank top5 candidates admitted at 10 bps, expected net return averages +0.9832%
versus a realized fixed-five-session conditional mean of -0.0636%. Rejected
candidates average -0.2769%. These overlap-in-time label diagnostics are not
portfolio returns and exclude unknown outcomes only from measurement, not selection.

The predeclared second block adds causal IMOEX returns over 5/21 sessions and
20-session log-return volatility to both models. These restore some market-level
information removed by cross-sectional feature normalization. At 10 bps:

| Policy | Market-augmented price-only return |
| --- | ---: |
| Rank, always invested | -7.52% |
| Absolute return, always invested | -27.78% |
| Absolute return + cash gate | -21.36% |
| Rank + absolute cash gate | -5.65% |

Augmented rank/absolute IC is 0.09247/0.05792; absolute MSE is 2.14% worse than the
historical-mean reference. The simple market-feature block did not resolve the
calibration issue. The verified AGRO/FIXP custody scenario gives identical final
returns for all variants; original quotes remain the primary scenario.

## Reproduce

Use source at the commit linked in the result archive and unpack the archive into
`experiments/asof_baseline/`. The original frozen `IMOEX.csv` belongs at
`experiments/ssa_targets_20261109/raw/IMOEX.csv`; the archive also supplies it under
`rank_return_results/frozen_inputs/IMOEX.csv` for use with the explicit --market flag.
The plan records hashes of the original source paths; restore that source path
before execution if using a fresh checkout.

```bash
python experiments/asof_baseline/compare_rank_return.py
python experiments/asof_baseline/run_rank_return.py
python experiments/asof_baseline/run_rank_return.py --custody
python -m unittest discover -s experiments/asof_baseline -p 'test_*.py'
```

Saved outputs include every forecast, fit boundary, inner score, trade, daily NAV,
calibration table, monthly/yearly comparison, both source scenarios and hashes.
95 focused tests pass. Seven new execution checks cover cash gates and allocation;
model tests cover chronology, market-feature causality and exact fee conversion.
Independent review found no current leakage/accounting blocker.

## Decision and limits

Retain rank as the comparison baseline. Absolute-return regression in this form
is not a replacement. Keep the hybrid as an unconfirmed hypothesis; its positive
aggregate delta does not justify selecting or tuning thresholds on this period.
A next research hypothesis is better calibrated return magnitudes and separating
market direction from relative stock selection, with clean future verification.

These are reused development periods. The already inspected May–October 2026
period was not used for new selection. Dividends remain incomplete and excluded;
price models can mistake dividend gaps for economic losses. Cash earns no interest.
The universe is retrospective; training excludes unavailable completed outcomes.
Fills, netting, fractional shares and costs are idealized; capacity, lot sizes and
tax are absent. No positive net total-return strategy is established.
