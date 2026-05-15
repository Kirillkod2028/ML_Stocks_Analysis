# EDA Report — ML_Stocks_Analysis
Generated: 2026-05-15 01:53

## Summary
- Total files: 45
- Groups: market (1), sectors (9), tickers (35)
- Columns per dataset: 50
- Date range market: 2022-08-01 → 2026-05-06

## Key Findings

### CRITICAL
| Finding | Detail | Action |
|---------|--------|--------|
| `Beta_USDRUB_60d` = constant 0.0 in market/sector | unique=1, std=0, zero signal | DROP from all datasets |
| `Beta_USDRUB_60d` = 100% NaN in all 35 ticker datasets | No values exist | DROP |

### MODERATE
| Finding | Detail | Action |
|---------|--------|--------|
| `dte_ex_days` 100% NaN for non-dividend companies | AGRO, FEES, FIXP, OZON, PIKK pay no dividends | Fill with 999 (sentinel) |
| Tickers T (212 rows) and YDEX (311 rows) | New listings 2025, R12_1/dist_52w_high 54-77% NaN | Keep, flag in metadata |
| AGRO data ends 2024-11-25 | Possible delisting | Keep in train, note in metadata |
| FIXP data ends 2025-06-13 | Trading suspension | Keep in train, note in metadata |
| 5 rows with target=1.0 | Max clip hit (legitimate) | Keep as-is (already clipped) |
| NaN in R12_1, R126, dist_52w_high | Indicator warm-up (first ~252 trading days) | Keep (imputer handles in pipeline) |

### BENIGN
| Finding | Detail |
|---------|--------|
| Duplicates | 0 in all 45 datasets |
| Data leakage | None detected (no future-named columns, target masked +/-7d of ex-div) |
| `post_ex_flag_10d` z-score >10 | Binary sparse feature, expected |
| Market ticker count 27-34/day | Variation due to new listings, expected |

## Per-Dataset Detail

### [market] market
- Rows: 31190 | Cols: 50 | NaN: 0.11% | Dups: 0
- Dates: 2022-08-01 to 2026-05-06
- Target: mean=0.0346 std=0.0377 pos%=98.1%
- NaN cols: `R12_1`=548(1.8%), `dist_52w_high`=545(1.7%), `R126`=193(0.6%), `vol_of_vol_60d`=101(0.3%), `R63`=71(0.2%), `TSMOM`=71(0.2%), `Beta_IMOEX_60d`=63(0.2%), `IVOL_60d`=63(0.2%)
- Outliers (|z|>10): `target`=14

### [sectors] BUILD
- Rows: 1911 | Cols: 50 | NaN: 0.00% | Dups: 0
- Dates: 2022-08-01 to 2026-05-06
- Target: mean=0.0428 std=0.0481 pos%=98.0%
- NaN cols: none
- Outliers (|z|>10): `post_ex_flag_10d`=9

### [sectors] ENERGY
- Rows: 2835 | Cols: 50 | NaN: 0.00% | Dups: 0
- Dates: 2022-08-01 to 2026-05-06
- Target: mean=0.0302 std=0.0274 pos%=98.6%
- NaN cols: none
- Outliers (|z|>10): none

### [sectors] IT
- Rows: 1347 | Cols: 50 | NaN: 0.05% | Dups: 0
- Dates: 2022-08-01 to 2026-05-06
- Target: mean=0.0372 std=0.0345 pos%=98.5%
- NaN cols: `TSMOM`=33(2.4%)
- Outliers (|z|>10): none

### [sectors] MOEXCN
- Rows: 2600 | Cols: 50 | NaN: 0.00% | Dups: 0
- Dates: 2022-08-01 to 2026-05-06
- Target: mean=0.0390 std=0.0398 pos%=98.3%
- NaN cols: none
- Outliers (|z|>10): `post_ex_flag_10d`=23

### [sectors] MOEXFN
- Rows: 4087 | Cols: 50 | NaN: 0.37% | Dups: 0
- Dates: 2022-08-01 to 2026-05-06
- Target: mean=0.0349 std=0.0504 pos%=98.7%
- NaN cols: `R12_1`=217(5.3%), `dist_52w_high`=216(5.3%), `R126`=97(2.4%), `vol_of_vol_60d`=53(1.3%), `R63`=38(0.9%), `TSMOM`=38(0.9%), `Beta_IMOEX_60d`=34(0.8%), `IVOL_60d`=34(0.8%)
- Outliers (|z|>10): `target`=5

### [sectors] MOEXMM
- Rows: 5674 | Cols: 50 | NaN: 0.00% | Dups: 0
- Dates: 2022-08-01 to 2026-05-06
- Target: mean=0.0346 std=0.0318 pos%=97.9%
- NaN cols: none
- Outliers (|z|>10): none

### [sectors] MOEXOG
- Rows: 7461 | Cols: 50 | NaN: 0.00% | Dups: 0
- Dates: 2022-08-01 to 2026-05-06
- Target: mean=0.0319 std=0.0340 pos%=97.8%
- NaN cols: none
- Outliers (|z|>10): `target`=4

### [sectors] MOEXTL
- Rows: 1886 | Cols: 50 | NaN: 0.00% | Dups: 0
- Dates: 2022-08-01 to 2026-05-06
- Target: mean=0.0301 std=0.0299 pos%=98.4%
- NaN cols: none
- Outliers (|z|>10): none

### [sectors] OTHER
- Rows: 3389 | Cols: 50 | NaN: 0.00% | Dups: 0
- Dates: 2022-08-01 to 2026-05-06
- Target: mean=0.0374 std=0.0393 pos%=97.6%
- NaN cols: none
- Outliers (|z|>10): `target`=1

### [tickers] AFLT
- Rows: 894 | Cols: 50 | NaN: 2.19% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0373 std=0.0370 pos%=97.5%
- NaN cols: `Beta_USDRUB_60d`=894(100.0%), `dte_ex_days`=86(9.6%)
- Outliers (|z|>10): `post_ex_flag_10d`=7

### [tickers] AGRO
- Rows: 534 | Cols: 50 | NaN: 4.00% | Dups: 0
- Dates: 2022-10-21 to 2024-11-25
- Target: mean=0.0441 std=0.0398 pos%=97.8%
- NaN cols: `Beta_USDRUB_60d`=534(100.0%), `dte_ex_days`=534(100.0%)
- Outliers (|z|>10): none

### [tickers] ALRS
- Rows: 884 | Cols: 50 | NaN: 2.62% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0298 std=0.0262 pos%=98.1%
- NaN cols: `Beta_USDRUB_60d`=884(100.0%), `dte_ex_days`=273(30.9%)
- Outliers (|z|>10): none

### [tickers] BSPB
- Rows: 871 | Cols: 50 | NaN: 2.31% | Dups: 0
- Dates: 2022-10-28 to 2026-05-06
- Target: mean=0.0416 std=0.0506 pos%=98.4%
- NaN cols: `Beta_USDRUB_60d`=871(100.0%), `dte_ex_days`=137(15.7%)
- Outliers (|z|>10): none

### [tickers] CHMF
- Rows: 885 | Cols: 50 | NaN: 2.52% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0340 std=0.0317 pos%=98.1%
- NaN cols: `Beta_USDRUB_60d`=885(100.0%), `dte_ex_days`=232(26.2%)
- Outliers (|z|>10): none

### [tickers] FEES
- Rows: 885 | Cols: 50 | NaN: 4.00% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0363 std=0.0305 pos%=98.6%
- NaN cols: `Beta_USDRUB_60d`=885(100.0%), `dte_ex_days`=885(100.0%)
- Outliers (|z|>10): none

### [tickers] FIXP
- Rows: 672 | Cols: 50 | NaN: 4.00% | Dups: 0
- Dates: 2022-10-21 to 2025-06-13
- Target: mean=0.0385 std=0.0451 pos%=98.7%
- NaN cols: `Beta_USDRUB_60d`=672(100.0%), `dte_ex_days`=672(100.0%)
- Outliers (|z|>10): none

### [tickers] FLOT
- Rows: 886 | Cols: 50 | NaN: 2.76% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0432 std=0.0482 pos%=97.9%
- NaN cols: `Beta_USDRUB_60d`=886(100.0%), `dte_ex_days`=338(38.1%)
- Outliers (|z|>10): none

### [tickers] GAZP
- Rows: 894 | Cols: 50 | NaN: 3.77% | Dups: 0
- Dates: 2022-10-28 to 2026-05-06
- Target: mean=0.0300 std=0.0337 pos%=97.5%
- NaN cols: `Beta_USDRUB_60d`=894(100.0%), `dte_ex_days`=790(88.4%)
- Outliers (|z|>10): none

### [tickers] GMKN
- Rows: 890 | Cols: 50 | NaN: 3.07% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0313 std=0.0275 pos%=98.1%
- NaN cols: `Beta_USDRUB_60d`=890(100.0%), `dte_ex_days`=477(53.6%)
- Outliers (|z|>10): `post_ex_flag_10d`=7

### [tickers] HYDR
- Rows: 889 | Cols: 50 | NaN: 2.20% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0290 std=0.0265 pos%=99.1%
- NaN cols: `Beta_USDRUB_60d`=889(100.0%), `dte_ex_days`=90(10.1%)
- Outliers (|z|>10): none

### [tickers] IRAO
- Rows: 884 | Cols: 50 | NaN: 2.26% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0238 std=0.0209 pos%=98.4%
- NaN cols: `Beta_USDRUB_60d`=884(100.0%), `dte_ex_days`=113(12.8%)
- Outliers (|z|>10): none

### [tickers] LKOH
- Rows: 870 | Cols: 50 | NaN: 2.27% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0259 std=0.0256 pos%=97.7%
- NaN cols: `Beta_USDRUB_60d`=870(100.0%), `dte_ex_days`=117(13.4%)
- Outliers (|z|>10): none

### [tickers] MAGN
- Rows: 889 | Cols: 50 | NaN: 2.62% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0336 std=0.0314 pos%=97.9%
- NaN cols: `Beta_USDRUB_60d`=889(100.0%), `dte_ex_days`=275(30.9%)
- Outliers (|z|>10): none

### [tickers] MGNT
- Rows: 887 | Cols: 50 | NaN: 2.49% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0319 std=0.0310 pos%=98.6%
- NaN cols: `Beta_USDRUB_60d`=887(100.0%), `dte_ex_days`=218(24.6%)
- Outliers (|z|>10): none

### [tickers] MOEX
- Rows: 882 | Cols: 50 | NaN: 2.21% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0308 std=0.0291 pos%=98.8%
- NaN cols: `Beta_USDRUB_60d`=882(100.0%), `dte_ex_days`=91(10.3%)
- Outliers (|z|>10): none

### [tickers] MTSS
- Rows: 884 | Cols: 50 | NaN: 2.21% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0263 std=0.0265 pos%=98.5%
- NaN cols: `Beta_USDRUB_60d`=884(100.0%), `dte_ex_days`=94(10.6%)
- Outliers (|z|>10): none

### [tickers] NLMK
- Rows: 890 | Cols: 50 | NaN: 2.85% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0349 std=0.0323 pos%=98.8%
- NaN cols: `Beta_USDRUB_60d`=890(100.0%), `dte_ex_days`=377(42.4%)
- Outliers (|z|>10): none

### [tickers] NVTK
- Rows: 869 | Cols: 50 | NaN: 2.32% | Dups: 0
- Dates: 2022-10-28 to 2026-05-06
- Target: mean=0.0337 std=0.0348 pos%=97.9%
- NaN cols: `Beta_USDRUB_60d`=869(100.0%), `dte_ex_days`=141(16.2%)
- Outliers (|z|>10): none

### [tickers] OZON
- Rows: 864 | Cols: 50 | NaN: 4.00% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0423 std=0.0387 pos%=98.0%
- NaN cols: `Beta_USDRUB_60d`=864(100.0%), `dte_ex_days`=864(100.0%)
- Outliers (|z|>10): none

### [tickers] PHOR
- Rows: 839 | Cols: 50 | NaN: 2.22% | Dups: 0
- Dates: 2022-10-28 to 2026-05-06
- Target: mean=0.0239 std=0.0216 pos%=97.5%
- NaN cols: `Beta_USDRUB_60d`=839(100.0%), `dte_ex_days`=94(11.2%)
- Outliers (|z|>10): none

### [tickers] PIKK
- Rows: 899 | Cols: 50 | NaN: 4.00% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0430 std=0.0508 pos%=98.0%
- NaN cols: `Beta_USDRUB_60d`=899(100.0%), `dte_ex_days`=899(100.0%)
- Outliers (|z|>10): none

### [tickers] PLZL
- Rows: 882 | Cols: 50 | NaN: 2.32% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0369 std=0.0303 pos%=98.4%
- NaN cols: `Beta_USDRUB_60d`=882(100.0%), `dte_ex_days`=142(16.1%)
- Outliers (|z|>10): none

### [tickers] POSI
- Rows: 870 | Cols: 50 | NaN: 3.40% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0375 std=0.0328 pos%=98.4%
- NaN cols: `Beta_USDRUB_60d`=870(100.0%), `dte_ex_days`=378(43.4%), `R12_1`=115(13.2%), `dist_52w_high`=114(13.1%)
- Outliers (|z|>10): none

### [tickers] ROSN
- Rows: 871 | Cols: 50 | NaN: 2.19% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0300 std=0.0309 pos%=97.8%
- NaN cols: `Beta_USDRUB_60d`=871(100.0%), `dte_ex_days`=84(9.6%)
- Outliers (|z|>10): none

### [tickers] RTKM
- Rows: 884 | Cols: 50 | NaN: 2.15% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0339 std=0.0327 pos%=98.6%
- NaN cols: `Beta_USDRUB_60d`=884(100.0%), `dte_ex_days`=67(7.6%)
- Outliers (|z|>10): none

### [tickers] SBER
- Rows: 884 | Cols: 50 | NaN: 2.19% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0253 std=0.0264 pos%=99.1%
- NaN cols: `Beta_USDRUB_60d`=884(100.0%), `dte_ex_days`=85(9.6%)
- Outliers (|z|>10): none

### [tickers] SIBN
- Rows: 869 | Cols: 50 | NaN: 2.21% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0305 std=0.0276 pos%=98.4%
- NaN cols: `Beta_USDRUB_60d`=869(100.0%), `dte_ex_days`=93(10.7%)
- Outliers (|z|>10): none

### [tickers] SMLT
- Rows: 894 | Cols: 50 | NaN: 3.77% | Dups: 0
- Dates: 2022-10-28 to 2026-05-06
- Target: mean=0.0406 std=0.0449 pos%=98.4%
- NaN cols: `Beta_USDRUB_60d`=894(100.0%), `dte_ex_days`=789(88.3%)
- Outliers (|z|>10): none

### [tickers] SNGS
- Rows: 884 | Cols: 50 | NaN: 2.19% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0345 std=0.0302 pos%=98.8%
- NaN cols: `Beta_USDRUB_60d`=884(100.0%), `dte_ex_days`=86(9.7%)
- Outliers (|z|>10): none

### [tickers] T
- Rows: 212 | Cols: 50 | NaN: 6.27% | Dups: 0
- Dates: 2025-06-24 to 2026-05-06
- Target: mean=0.0209 std=0.0199 pos%=99.5%
- NaN cols: `Beta_USDRUB_60d`=212(100.0%), `R12_1`=164(77.4%), `dist_52w_high`=163(76.9%), `dte_ex_days`=82(38.7%), `R126`=44(20.8%)
- Outliers (|z|>10): none

### [tickers] TATN
- Rows: 852 | Cols: 50 | NaN: 2.28% | Dups: 0
- Dates: 2022-10-28 to 2026-05-06
- Target: mean=0.0313 std=0.0321 pos%=97.7%
- NaN cols: `Beta_USDRUB_60d`=852(100.0%), `dte_ex_days`=118(13.8%)
- Outliers (|z|>10): none

### [tickers] TRNFP
- Rows: 880 | Cols: 50 | NaN: 2.20% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0308 std=0.0357 pos%=98.5%
- NaN cols: `Beta_USDRUB_60d`=880(100.0%), `dte_ex_days`=86(9.8%)
- Outliers (|z|>10): none

### [tickers] VTBR
- Rows: 890 | Cols: 50 | NaN: 2.20% | Dups: 0
- Dates: 2022-10-21 to 2026-05-06
- Target: mean=0.0413 std=0.0821 pos%=98.5%
- NaN cols: `Beta_USDRUB_60d`=890(100.0%), `dte_ex_days`=90(10.1%)
- Outliers (|z|>10): `R1`=1, `R5`=1, `VWAP_dev_5d`=1, `post_ex_flag_10d`=7, `target`=5

### [tickers] YDEX
- Rows: 311 | Cols: 50 | NaN: 5.37% | Dups: 0
- Dates: 2025-02-06 to 2026-05-06
- Target: mean=0.0292 std=0.0245 pos%=98.7%
- NaN cols: `Beta_USDRUB_60d`=311(100.0%), `R12_1`=168(54.0%), `dist_52w_high`=167(53.7%), `dte_ex_days`=141(45.3%), `R126`=48(15.4%)
- Outliers (|z|>10): none

## Summary Table

| Group | Name | Rows | NaN% | Dups | Date Range |
|-------|------|------|------|------|------------|
| market | market | 31190 | 0.11 | 0 | 2022-08-01 to 2026-05-06 |
| sectors | BUILD | 1911 | 0.0 | 0 | 2022-08-01 to 2026-05-06 |
| sectors | ENERGY | 2835 | 0.0 | 0 | 2022-08-01 to 2026-05-06 |
| sectors | IT | 1347 | 0.049 | 0 | 2022-08-01 to 2026-05-06 |
| sectors | MOEXCN | 2600 | 0.0 | 0 | 2022-08-01 to 2026-05-06 |
| sectors | MOEXFN | 4087 | 0.372 | 0 | 2022-08-01 to 2026-05-06 |
| sectors | MOEXMM | 5674 | 0.0 | 0 | 2022-08-01 to 2026-05-06 |
| sectors | MOEXOG | 7461 | 0.0 | 0 | 2022-08-01 to 2026-05-06 |
| sectors | MOEXTL | 1886 | 0.0 | 0 | 2022-08-01 to 2026-05-06 |
| sectors | OTHER | 3389 | 0.0 | 0 | 2022-08-01 to 2026-05-06 |
| tickers | AFLT | 894 | 2.192 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | AGRO | 534 | 4.0 | 0 | 2022-10-21 to 2024-11-25 |
| tickers | ALRS | 884 | 2.618 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | BSPB | 871 | 2.315 | 0 | 2022-10-28 to 2026-05-06 |
| tickers | CHMF | 885 | 2.524 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | FEES | 885 | 4.0 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | FIXP | 672 | 4.0 | 0 | 2022-10-21 to 2025-06-13 |
| tickers | FLOT | 886 | 2.763 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | GAZP | 894 | 3.767 | 0 | 2022-10-28 to 2026-05-06 |
| tickers | GMKN | 890 | 3.072 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | HYDR | 889 | 2.202 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | IRAO | 884 | 2.256 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | LKOH | 870 | 2.269 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | MAGN | 889 | 2.619 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | MGNT | 887 | 2.492 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | MOEX | 882 | 2.206 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | MTSS | 884 | 2.213 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | NLMK | 890 | 2.847 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | NVTK | 869 | 2.325 | 0 | 2022-10-28 to 2026-05-06 |
| tickers | OZON | 864 | 4.0 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | PHOR | 839 | 2.224 | 0 | 2022-10-28 to 2026-05-06 |
| tickers | PIKK | 899 | 4.0 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | PLZL | 882 | 2.322 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | POSI | 870 | 3.395 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | ROSN | 871 | 2.193 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | RTKM | 884 | 2.152 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | SBER | 884 | 2.192 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | SIBN | 869 | 2.214 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | SMLT | 894 | 3.765 | 0 | 2022-10-28 to 2026-05-06 |
| tickers | SNGS | 884 | 2.195 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | T | 212 | 6.274 | 0 | 2025-06-24 to 2026-05-06 |
| tickers | TATN | 852 | 2.277 | 0 | 2022-10-28 to 2026-05-06 |
| tickers | TRNFP | 880 | 2.195 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | VTBR | 890 | 2.202 | 0 | 2022-10-21 to 2026-05-06 |
| tickers | YDEX | 311 | 5.37 | 0 | 2025-02-06 to 2026-05-06 |