# Conditional history of AGRO and FIXP holdings

Checked on 2026-11-10. The existing 35-ticker sample ends the observed TQBR
quotes of AGRO on 2024-12-02 and FIXP on 2025-06-20. These are obsolete receipt
codes, not evidence that the related business lost all economic value. Their
successors are distinct legal securities; replacement is conditional on custody
and successful allocation/exchange.

| Old security | Successor security | Ratio: new shares per old GDR | First regular tradable session |
| --- | --- | ---: | --- |
| AGRO, US7496552057 | RAGR, RU000A0JQUZ6 | 7.0135 | 2025-02-17 |
| FIXP, US33835G2057 | FIXR, RU000A10B5G8 | 158 | 2025-08-20 |

## Verified evidence and remaining assumptions

**AGRO.** [MOEX](https://www.moex.com/n77722) confirms RAGR identity and the
17 February 2025 trading start. The [issuer's own investor post](https://www.tbank.ru/invest/social/profile/Rusagro/25b9175a-f89d-412f-b752-b71ef66f5ee3/)
confirms the exact 7.0135 allocation and account-level rounding to whole shares;
using a rounded ratio of 7 changes economic value. The [depository notice](https://bcs.ru/newsdepo/1024099)
reproduces NSD instructions for eligible accounts and a 9 December 2024 record.
This was a direct-ownership distribution under the Russian significant-company
procedure, not a blanket ticker rename. [Issuer instructions](https://www.rusagrogroup.ru/ru/investoram/shares-distribution/)
describe a separate application route for some foreign-custody holders.
The simulated scenario therefore assumes an eligible Russian-depository holding
whose shares were actually credited before the first trading day.

**FIXP.** The [5 June issuer announcement](https://ir.fix-price.com/upload/iblock/22f/27rf3pxxb3qjfztprwn375cd44xgse9y/Press%20release_Exchange_ENG.pdf)
sets the voluntary offer ratio to 158 and differentiates exchange and OTC routes.
Nonparticipants retain GDR ownership without access to organized Russian trading.
The scenario assumes timely accepted exchange-route participation through Russian
custody. The planned exchange settlement was 10 July; broker-level share-crediting
dates could vary. The [completion release](https://media.fix-price.com/news/fix-price-obyavlyaet-itogi-obmena-gdr-na-aktsii-pao-fiks-prays/?back=8786)
confirms completion for most accepted offers and the actual 20 August trading
start. The initially expected 6 August start was postponed: use the final
[MOEX notice](https://www.moex.com/n92917), not the indicative prospectus date.
The contemplated 1000:1 FIXR consolidation is not applied: [MOEX](https://www.moex.com/n97730)
postponed the conversion in February 2026, and collected observations have no
corresponding unit jump through 6 November 2026.

## Execution-only scenario

`recover_security_history.py` downloads paginated TQBR observations for RAGR and
FIXR. It independently fetches ISS security metadata and checks actual SECID and
ISIN against the verified ledger before annotating the history. Every response is
retained with a hash and URL in `continuity_data/manifest.json`.

```bash
python experiments/asof_baseline/recover_security_history.py --acknowledge AGRO FIXP
```

For reproduction from the archive, use the frozen inputs without network access:

```bash
python experiments/asof_baseline/recover_security_history.py --offline --acknowledge AGRO FIXP
```

Offline replay verifies SHA-256 hashes of the baseline quotes, successor CSVs,
security-identity JSON and every retained ISS page. It also requires the current
transition ledger, custody acknowledgements and date boundary to match the
frozen manifest. A mismatch aborts before output changes. The download manifest
is preserved; an identical derived quote grid is left untouched. Online mode is
an explicit refresh and may observe subsequently revised market history.

The new `quotes_conversion_scenario.parquet` expresses each successor quote in
units of one old receipt (`successor price * shares per receipt`). This is
equivalent to converting an existing position's share count for valuation and
liquidation. Original quotes and every non-trading interval remain unchanged;
missing successor observations remain missing. The routine rejects overlapping
old/new observed prices and mismatched identifiers. It does not fabricate a new
prediction, add successor names to a historical stock universe, or rerank models.
Only the explicitly acknowledged cases are bridged; the unchanged quote grid is
the conservative unresolved scenario.

Bridged rows carry `entry_allowed=False`: successor prices may value and sell an
existing receipt holding, but cannot fund a new purchase under the obsolete
ticker. Portfolio simulators enforce this field. All four stored prediction
files contain zero AGRO forecasts from 17 February 2025 and zero FIXP forecasts
from 20 August 2025; the additional guard does not change the frozen experiment.

This calculation assumes fractional shares, zero conversion fees and no separate
residual old-receipt claim. Actual rounding, broker-specific credit delays, taxes,
restrictions and (for OTC FIXP) cash fees require account-level data. A successful
quote download does not prove that a particular holder completed an exchange.
The scenario is not a full point-in-time security master.

## Same-ticker legal-security change: OZON

A stable ticker is not proof of stable legal security. [MOEX n93976](https://www.moex.com/n93976)
records replacement of OZON ADS `US69269L1044` by ordinary shares
`RU000A10CW95` on 29 September 2025; [n95178](https://www.moex.com/n95178)
confirms actual trading resumption on 11 November. [n93647](https://www.moex.com/n93647)
specifies the last T+1 session as 22 September (other settlement modes had
different restrictions); the TQBR quote gap starts on 23 September. The [issuer chronology and
conversion FAQ](https://ir.ozon.com/redomiciliation) confirms one share per ADS
and completion of automatic conversion on 3–20 October. The [NSD notice](https://www.nsd.ru/ru/news/o-publikatsii-daty-sostavleniya-spiska-depozitariev-osuschestvlyayuschih-uchet-prav-na-amerikanskie-depozitarnye-raspiski-na-obyknovennye-aktsii-ozon-holdings-plc-isin-us69269l1044-262654/)
identifies 3 October as the depository-list date for the automatic procedure.
Foreign-custody holders had a separate application process.

Both the original and the AGRO/FIXP-conversion scenario therefore assume a
completed Russian-custody 1:1 conversion for OZON positions held across the pause.
This is explicit in `SAME_TICKER_TRANSITIONS` and the recovery manifest. The
daily top5 and daily buffer experiment sell OZON on 17 September, before the
pause; five-session phases 1 and 3 hold it through the pause and sell on
11 November. The equal-universe cohort reference also has crossing holdings.
Those results are conditional on completed conversion, not an unconditional
foreign-custody scenario. Missing opens during the pause remain missing.

## Recovered same-ISIN alias: TCSG → T

[MOEX n74886](https://www.moex.com/n74886) announced on 14 November 2024 that
TCSG changes its trading code to T on 28 November. The issue registration stays
`1-01-16784-A`, ISIN `RU000A107UL4`. [n68303](https://www.moex.com/n68303)
confirms the ordinary share began trading on 18 March 2024. Earlier TCSG rows
cannot be silently appended: [n67851](https://www.moex.com/n67851) records the
change from GDR `US87238U2033` to this ordinary share on 27 February 2024.

The separate recovery contains **181 observed sessions, 18 March–27 November
2024**, with ISS page snapshots, identity verification and SHA-256 hashes:

```bash
python experiments/asof_baseline/recover_security_history.py --aliases-only
```

`continuity_data/aliases/TCSG_same_isin.csv` preserves the actual historical
`TCSG` code and raw, unadjusted share prices. It records the announcement time
and successor identifier, and rejects dates outside the verified same-security
interval. This restores data for a later panel rebuild; current forecasts and
portfolio comparisons remain fixed. When integrated, the 2026 T split must be
applied exactly once to this raw segment through the ordinary split ledger.
Pre-March-2024 TCSG receipts and other reorganizations remain outside this
verified alias segment; this is not a complete historical security master.
