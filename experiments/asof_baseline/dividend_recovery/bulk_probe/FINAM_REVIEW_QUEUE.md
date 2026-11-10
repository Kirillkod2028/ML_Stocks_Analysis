# Broader dividend recovery queue

On 2026-11-10 all 33 remaining project tickers were probed at the public Finam
broker dividend-history URL. The web extraction exposed a historical table
on 32 pages. There are **203 unverified candidates for 31 matching tickers**
within record dates 2021-01-01 through 2026-11-10. Four additional rows from
the FIXP URL are quarantined: it redirects to FIXR, a different security.
The AGRO URL is inaccessible. No new verified event or coverage interval
is created by this collection.

This is useful recoverable data, not a blanket access failure. Finam's
extracted history page exposes more than BCS's five current rows: for example,
CHMF provides eight candidate events in this interval and ROSN eleven.
Twenty-seven visible tables reach before the interval's start. That temporal
reach does not prove issuer completeness. Twelve candidate rows belong to
same-date groups and remain separate, including CHMF's 191.51 and 38.30 RUB
distributions for record date 2024-06-18.

The source snapshot contains focused numerical/date table rows, per-page
URLs, references, identity checks, access failures and observation date.
`collect_finam_candidates.py` replays it into `finam_candidates.csv`,
`finam_quarantined.csv`, `finam_access_audit.csv` and `finam_summary.json`:

```bash
python experiments/asof_baseline/collect_finam_candidates.py
python experiments/asof_baseline/test_finam_candidates.py
```

Direct HTTP requests for CHMF, ROSN, MTSS and GAZP returned 403. BCS direct
HTTP returned a JavaScript challenge despite status 200. Web extraction of
the public pages succeeded. Therefore this replay is reproducible from the
saved snapshot, but a reliable unattended HTTP refresh is not established.
No challenge or authorization control was bypassed.

The table columns identify last-cum date, record date and displayed cash.
They do not supply verified first ex-session, historical public announcement
time, actual customer payment date, or a proven share unit. All candidate
rows have `price_basis=unknown`, `ex_date` empty, and unverified status.
Corporate conversions and possible provider normalization require review
before amounts can join the price panel. Do not treat a page absence as
zero dividends, do not copy candidate rows into the reviewed ledger, and do
not feed future outcomes into features.

Next concrete step: reconcile candidate amounts and all events against each
issuer's history, verify last-cum/ex-session evidence and share units, then
add explicit reviewed coverage. Finam's AGRO alternative is
[RAGR](https://www.finam.ru/quote/moex/ragr/dividends/), while the issuer has
[a dividend history](https://www.rusagrogroup.ru/ru/investoram/dividendy/).
Those are leads for an identity/conversion review, not permission to stitch
RAGR data onto AGRO. The original FIXP URL likewise resolves to
[FIXR](https://www.finam.ru/quote/moex/fixr/dividends/).

A cross-check against the separately collected BCS queue matched 104 observed
date/cash rows and found eight apparent disagreements, preserved in
`finam_bcs_disagreements.csv`. GMKN and TRNFP histories differ by a factor
of 100 for pre-conversion events. For POSI on 2024-05-24, BCS displays 51.89
while Finam has separate 47.33 and 4.56 distributions that sum to 51.89.
FEES, HYDR and MAGN differ in displayed precision. These comparisons do not
choose a correct source; they identify the share-unit, event-identity and
precision reviews required before admission to the verified ledger.
