# Dividend label contract and available sources

Checked on 2026-11-10. A complete dividend ledger for the 35-ticker universe
has **not** been recovered. Existing development scores remain price-return
scores. The accounting function does not make that history complete. The reviewed
fallback now contains 88 events for 11 issuers with explicit bounded intervals
(nine through 2026-11-06, PHOR/HYDR through 2025-12-31). See
[recovered evidence](dividend_recovery/README.md); uncovered values remain NaN.

Follow-up: 16 events for SBER/LKOH have now been independently reviewed and
applied to 2,166 panel labels. The other 33 tickers remain uncovered. See
[reviewed recovery](dividend_recovery/README.md) for evidence and limitations.

## Source checks

- Public `https://iss.moex.com/iss/securities/SBER/dividends.json` responds
  HTTP 200 with `description` and `boards`, without a `dividends` table.
  The same result was observed for `SBER03` (historical security alias) and
  `MAGN`. HTTP success is therefore not evidence of available payouts.
- `https://iss.moex.com/iss/rms/engines/futures/objects/cashflow.json?date=2024-07-01`
  does return a cashflow table. It is **not a realized-dividend ledger**:
  the official [schema announcement](https://www.moex.com/n56498) describes
  `CF` as anticipated dividends used for derivatives. The historical response
  includes GAZR forecasts for 2025, 2026 and 2027. Do not convert these to
  observed payouts or use their dates as verified stock ex-dates.
- [API NSD](https://nsddata.ru/ru/products/2) offers `getCorpActions V.2`
  and identifiers joining actions to securities. MOEX's
  [corporate-data subscription page](https://www.moex.com/ru/forms/referencedata.aspx)
  describes ordering access to past and future corporate actions. No
  credentials or contract for this feed are present in the current project.
- Issuer dividend history is a possible independently reviewed fallback:
  [Sberbank](https://www.sberbank.com/investor-relations/dividends),
  [LUKOIL](https://lukoil.ru/InvestorAndShareholderCenter/Securities/Dividends).
  Record dates and cash amounts alone do not establish a complete usable
  event ledger with first ex-dates for every ticker and period.

The official MOEX [T+1 presentation](https://www.moex.com/files/4vhzx0fnqzx0sr51amkgx1s1d0)
describes the transition from T+2 on 2023-07-31. Ex-dates depend on actual
settlement eligibility and calendars. Weekend/holiday dates, suspensions and
additional trading sessions prevent replacing verification with a generic
weekday subtraction. The [total-return index notice](https://www.moex.com/n63176)
also states the special record-date/weekend convention used by the exchange.

## Accounting function

`ml_stocks_analysis.corporate_actions.total_return_labels(labels, events, coverage)`
requires these input columns:

| Table | Required columns |
| --- | --- |
| labels | ticker, entry_date, exit_date, entry_price, exit_price, price_basis |
| events | ticker, event_id, ex_date, cash_amount, currency, price_basis, source_url, verified_at |
| coverage | ticker, start_date, end_date, source_url, verified_at |

Each coverage row is an explicit assertion, established by reviewing the
source, that **all** events for that ticker in that inclusive calendar-date
interval are present. Unknown coverage should be omitted. A partial list of
known dividends cannot certify absence of other dividends.

For an open-entry / close-exit position, entitlement is
`entry_date < ex_date <= exit_date`. Gross label:
`(exit_price + sum(entitled_cash_amounts)) / entry_price - 1`.
Uncovered labels return `dividends_known=False`, `cash_dividends=NaN` and
`total_return=NaN`; price returns remain available. An empty event table implies
zero only inside independently verified complete coverage.

`price_basis` must identify the **same split-adjusted share unit** for both
prices and dividend amounts. The caller must apply the verified split ledger
before using this function. It is an explicit contract, not an inferred
adjustment. The current interface accepts RUB only and no tax deduction.

The outcome recognizes a dividend receivable at ex-date. It must not be used
to make that receivable immediately spendable: an investable portfolio needs
actual payment dates, tax and settlement accounting. `verified_at` is the
date the dataset was checked, not an announcement timestamp. Future events
must not become predictive features just because they exist in this ledger.

Focused checks: `python experiments/asof_baseline/test_corporate_actions.py`.
