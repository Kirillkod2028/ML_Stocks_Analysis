#!/usr/bin/env python3
"""
build_datasets.py
=================
Сборщик обучающего датасета для ML-модели прогноза max-5d роста акций MOEX.

Целевая переменная:
    target = max(ADJ_HIGH[t+1..t+5]) / ADJ_CLOSE[t] - 1

Источник данных: бесплатный ISS MOEX REST API (без авторизации).

Структура датасетов (parquet / snappy):
    datasets/market/dataset_market.parquet
    datasets/sectors/dataset_{SECTOR}.parquet
    datasets/tickers/dataset_{TICKER}.parquet

Запуск:
    pip install -r requirements.txt
    python build_datasets.py

Jupyter-ноутбук:
    await build_all_datasets_async()   # event loop уже запущен
    # или: asyncio.run(build_all_datasets_async())

Архитектура I/O:
    - asyncio + aiohttp: все HTTP-запросы к ISS асинхронны
    - asyncio.Semaphore(CONCURRENCY): ограничивает одновременные запросы к серверу
    - Бенчмарки (IMOEX, USDRUB, RVI): параллельно через asyncio.gather
    - Для каждого тикера: history + dividends параллельно через asyncio.gather
    - Все 35 тикеров: параллельно (ограничены семафором)
    - Расчёты pandas/numpy: синхронные (CPU-bound, GIL несущественен)

Оценка прироста скорости:
    Sync  : N_tickers × (N_pages × (latency + delay)) ≈ 35 × 12 × 0.75s ≈ 5.5 мин
    Async : ceil(N/CONCURRENCY) × (N_pages × delay)   ≈ 6  × 12 × 0.25s ≈  18 сек
    → 4-6× быстрее при CONCURRENCY=6
"""

import asyncio
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

import aiohttp
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

FETCH_START   = "2021-01-01"
DATA_START    = "2022-08-01"
END_DATE      = datetime.today().strftime("%Y-%m-%d")
OUTPUT_DIR    = Path("datasets")
NAN_THRESHOLD = 0.20

ISS_BASE      = "https://iss.moex.com/iss"
CONCURRENCY   = 6      # max одновременных запросов к ISS (семафор)
REQUEST_DELAY = 0.20   # пауза между страницами одного тикера (вежливость к ISS)
RETRY_DELAY   = 2.0    # базовая пауза при retry
TIMEOUT_SEC   = 45     # HTTP timeout

TICKERS_SECTORS: dict[str, str] = {
    "GAZP": "MOEXOG", "LKOH": "MOEXOG", "NVTK": "MOEXOG", "ROSN": "MOEXOG",
    "TATN": "MOEXOG", "SNGS": "MOEXOG", "SIBN": "MOEXOG", "TRNFP": "MOEXOG",
    "SBER": "MOEXFN", "VTBR": "MOEXFN", "T":    "MOEXFN",
    "MOEX": "MOEXFN", "BSPB": "MOEXFN",
    "GMKN": "MOEXMM", "CHMF": "MOEXMM", "NLMK": "MOEXMM",
    "MAGN": "MOEXMM", "ALRS": "MOEXMM", "PLZL": "MOEXMM",
    "MGNT": "MOEXCN", "OZON": "MOEXCN", "FIXP": "MOEXCN",
    "MTSS": "MOEXTL", "RTKM": "MOEXTL",
    "HYDR": "ENERGY", "IRAO": "ENERGY", "FEES": "ENERGY",
    "YDEX": "IT",     "POSI": "IT",
    "PIKK": "BUILD",  "SMLT": "BUILD",
    "AFLT": "OTHER",  "FLOT": "OTHER", "PHOR": "OTHER", "AGRO": "OTHER",
}

FEATURE_COLS: list[str] = [
    "R1", "R5", "R21", "R63", "R126", "R12_1",
    "TSMOM", "dist_52w_high", "price_accel",
    "Z_SMA20", "BB_pct_b", "RSI_14", "Williams_R_14", "VWAP_dev_5d",
    "RV20", "GK_vol_10d", "Parkinson_vol_10d", "RS_vol_10d", "YZ_vol_10d",
    "BB_BandWidth", "ATR_pct", "vol_of_vol_60d", "skew_20d", "kurt_20d",
    "V_zscore_20d", "Amihud_ILLIQ_20d", "OBV_slope",
    "NUMTRADES_zscore_20d", "ADV_ratio_5_60", "turnover_rel",
    "MACD_hist", "MACD_slope_sign", "Stoch_K_14", "ADX_14",
    "CCI_20", "EMA_cross_20_50", "Donchian_pos_20",
    "Beta_IMOEX_60d", "RS_vs_IMOEX_20d", "IVOL_60d",
    "Beta_USDRUB_60d", "RVI_level",
    "day_of_week", "month", "dte_ex_days", "post_ex_flag_10d",
]

CATEGORICAL_COLS: set[str] = {
    "TSMOM", "MACD_slope_sign", "EMA_cross_20_50",
    "day_of_week", "month", "post_ex_flag_10d",
}
NORM_COLS: list[str] = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC ISS CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

def _make_session() -> aiohttp.ClientSession:
    timeout   = aiohttp.ClientTimeout(total=TIMEOUT_SEC, connect=10)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY * 3, ssl=False)
    headers   = {"User-Agent": "MOEX-ML-DataCollector/2.0 (academic)"}
    return aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers)


async def _iss_get(
    session:  aiohttp.ClientSession,
    sem:      asyncio.Semaphore,
    url:      str,
    params:   dict | None = None,
    retries:  int = 3,
) -> dict | None:
    """
    Async GET к ISS с семафором и retry.

    Семафор гарантирует не более CONCURRENCY одновременных запросов к ISS.
    Страницы одного тикера последовательны (cursor-based) — их нельзя
    распараллелить внутри одного тикера. Параллелизм — между разными тикерами.
    """
    async with sem:
        for attempt in range(retries):
            try:
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == retries - 1:
                    logger.warning("ISS failed [%s]: %s",
                                   url.split("/")[-1].split("?")[0], exc)
                    return None
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
    return None


async def _paginate(
    session:     aiohttp.ClientSession,
    sem:         asyncio.Semaphore,
    url:         str,
    base_params: dict,
    section:     str = "history",
) -> tuple[list, list]:
    """
    Постраничный cursor-based сбор данных ISS.
    Страницы одного запроса последовательны (cursor из ответа).
    Задержка REQUEST_DELAY между страницами не блокирует другие корутины.
    """
    all_rows: list = []
    cols:     list = []
    cursor = 0

    while True:
        data = await _iss_get(session, sem, url, {**base_params, "start": cursor})
        if data is None:
            break
        sect = data.get(section, {})
        rows = sect.get("data", [])
        if not rows:
            break
        if not cols:
            cols = sect.get("columns", [])
        all_rows.extend(rows)

        c_info = data.get(f"{section}.cursor", {}).get("data", [])
        if c_info:
            total, pagesize = c_info[0][1], c_info[0][2]
            cursor += pagesize
            if cursor >= total:
                break
        else:
            break
        await asyncio.sleep(REQUEST_DELAY)

    return all_rows, cols


async def fetch_history(
    session: aiohttp.ClientSession,
    sem:     asyncio.Semaphore,
    ticker:  str,
    start:   str = FETCH_START,
    end:     str = END_DATE,
) -> pd.DataFrame:
    url    = (f"{ISS_BASE}/history/engines/stock/markets/shares"
              f"/boards/TQBR/securities/{ticker}.json")
    params = {"from": start, "till": end,
              "columns": "TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME,VALUE,WAPRICE,NUMTRADES"}
    rows, cols = await _paginate(session, sem, url, params)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE", "WAPRICE", "NUMTRADES"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["CLOSE"]).query("CLOSE > 0").sort_values("TRADEDATE")
    # ISS может вернуть несколько строк на одну дату (разные сессии/борды)
    df = df.drop_duplicates(subset=["TRADEDATE"], keep="last")
    df["WAPRICE"]   = df["WAPRICE"].fillna(df["CLOSE"])
    df["NUMTRADES"] = df["NUMTRADES"].fillna(0)
    return df.reset_index(drop=True)


async def fetch_dividends(
    session: aiohttp.ClientSession,
    sem:     asyncio.Semaphore,
    ticker:  str,
) -> pd.DataFrame:
    url  = f"{ISS_BASE}/securities/{ticker}/dividends.json"
    data = await _iss_get(session, sem, url)
    if data is None:
        return pd.DataFrame()
    rows = data.get("dividends", {}).get("data", [])
    cols = data.get("dividends", {}).get("columns", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)
    if "registryclosedate" in df.columns:
        df["registryclosedate"] = pd.to_datetime(df["registryclosedate"], errors="coerce")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "currencyid" in df.columns:
        df = df[df["currencyid"].isin(["RUB", "SUR"]) | df["currencyid"].isna()]
    return df.dropna(subset=["registryclosedate", "value"]).query("value > 0").reset_index(drop=True)


async def fetch_index(
    session:   aiohttp.ClientSession,
    sem:       asyncio.Semaphore,
    index_id:  str,
    start:     str = FETCH_START,
    end:       str = END_DATE,
) -> pd.DataFrame:
    url = f"{ISS_BASE}/history/engines/stock/markets/index/securities/{index_id}.json"
    rows, cols = await _paginate(session, sem, url,
                                  {"from": start, "till": end, "columns": "TRADEDATE,CLOSE"})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df["CLOSE"]     = pd.to_numeric(df["CLOSE"], errors="coerce")
    df = df.dropna(subset=["CLOSE"]).sort_values("TRADEDATE")
    df = df.drop_duplicates(subset=["TRADEDATE"], keep="last")
    return df.reset_index(drop=True)


async def fetch_usdrub(
    session: aiohttp.ClientSession,
    sem:     asyncio.Semaphore,
    start:   str = FETCH_START,
    end:     str = END_DATE,
) -> pd.DataFrame:
    url = f"{ISS_BASE}/history/engines/currency/markets/selt/securities/USD000UTSTOM.json"
    rows, cols = await _paginate(session, sem, url, {"from": start, "till": end})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    close_col = next((c for c in ["CLOSE", "WAPRICE", "LEGALCLOSEPRICE", "MARKETPRICE2"]
                      if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()),
                     None)
    if close_col is None:
        return pd.DataFrame()
    df = df[["TRADEDATE", close_col]].rename(columns={close_col: "USDRUB"})
    df["USDRUB"] = pd.to_numeric(df["USDRUB"], errors="coerce")
    df = df.dropna().sort_values("TRADEDATE")
    df = df.drop_duplicates(subset=["TRADEDATE"], keep="last")
    return df.reset_index(drop=True)


async def fetch_ticker_data(
    session: aiohttp.ClientSession,
    sem:     asyncio.Semaphore,
    ticker:  str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Параллельно загружает history + dividends одного тикера.
    Два независимых ISS-запроса через asyncio.gather.
    """
    raw, divs = await asyncio.gather(
        fetch_history(session, sem, ticker),
        fetch_dividends(session, sem, ticker),
    )
    return raw, divs


# ═══════════════════════════════════════════════════════════════════════════════
# DIVIDEND ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _get_ex_date(record_date: pd.Timestamp) -> pd.Timestamp:
    """registryclosedate → ex-dividend дата (T+1 после 24.07.2023, T+2 до)."""
    offset = (pd.offsets.BusinessDay(1) if record_date >= pd.Timestamp("2023-07-24")
              else pd.offsets.BusinessDay(2))
    return record_date - offset


def backward_adjust(df: pd.DataFrame, divs: pd.DataFrame) -> pd.DataFrame:
    """
    Мультипликативная backwards-корректировка OHLC + WAPRICE.
    Фактор считается по raw CLOSE (стандарт Bloomberg/Refinitiv).
    Алгоритм: от новейших дивидендов к старейшим.
    """
    df = df.copy().sort_values("TRADEDATE").reset_index(drop=True)
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "WAPRICE"]:
        df[f"ADJ_{col}"] = df[col].astype(float).copy()

    if divs.empty or "registryclosedate" not in divs.columns:
        return df

    valid = (divs[["registryclosedate", "value"]].dropna().query("value > 0").copy())
    valid["ex_date"] = valid["registryclosedate"].apply(_get_ex_date)
    valid = valid.sort_values("ex_date", ascending=False)

    for _, row in valid.iterrows():
        ex_date = row["ex_date"]
        D       = float(row["value"])
        pre     = df["TRADEDATE"] < ex_date
        if not pre.any():
            continue
        C_pre = float(df.loc[pre, "CLOSE"].iloc[-1])
        if C_pre <= 0 or np.isnan(C_pre) or D >= C_pre:
            continue
        factor = (C_pre - D) / C_pre
        for col in ["OPEN", "HIGH", "LOW", "CLOSE", "WAPRICE"]:
            df.loc[pre, f"ADJ_{col}"] = (df.loc[pre, f"ADJ_{col}"] * factor).values
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr   = pd.concat([high - low, (high - close.shift(1)).abs(),
                      (low - close.shift(1)).abs()], axis=1).max(axis=1)
    dh, dl = high.diff(), -low.diff()
    pdm  = pd.Series(np.where((dh > dl) & (dh > 0), dh, 0.0), index=close.index)
    mdm  = pd.Series(np.where((dl > dh) & (dl > 0), dl, 0.0), index=close.index)
    atr  = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    pdi  = 100 * pdm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    mdi  = 100 * mdm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def _yz_volatility(o, h, l, c, n=10) -> pd.Series:
    eps    = 1e-10
    lco    = np.log((o / c.shift(1)).clip(lower=eps))
    loc_   = np.log((c / o).clip(lower=eps))
    lho    = np.log((h / o).clip(lower=eps))
    llo    = np.log((l / o).clip(lower=eps))
    rs     = (lho*(lho - loc_) + llo*(llo - loc_)).rolling(n).mean()
    k      = 0.34 / (1.34 + (n+1)/max(n-1, 1))
    return np.sqrt((lco.rolling(n).var() + k*loc_.rolling(n).var() + (1-k)*rs
                    ).clip(lower=0) * 252)


def _rolling_beta(y: pd.Series, x: pd.Series, w: int = 60) -> pd.Series:
    return y.rolling(w).cov(x) / x.rolling(w).var().replace(0, np.nan)


def _rolling_ivol(y: pd.Series, x: pd.Series, w: int = 60) -> pd.Series:
    b = _rolling_beta(y, x, w)
    return np.sqrt((y.rolling(w).var() - b**2 * x.rolling(w).var()).clip(lower=0) * 252)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_features(
    df:         pd.DataFrame,
    imoex_ret:  pd.Series | None = None,
    usdrub_ret: pd.Series | None = None,
    rvi:        pd.Series | None = None,
) -> pd.DataFrame:
    df  = df.sort_values("TRADEDATE").drop_duplicates(subset=["TRADEDATE"], keep="last")
    df  = df.set_index("TRADEDATE").sort_index()
    eps = 1e-10
    o   = df["ADJ_OPEN"].replace(0, np.nan)
    h   = df["ADJ_HIGH"].replace(0, np.nan)
    l   = df["ADJ_LOW"].replace(0, np.nan)
    c   = df["ADJ_CLOSE"].replace(0, np.nan)
    vol = df["VOLUME"].fillna(0)
    val = df["VALUE"].fillna(0)
    wap = df["ADJ_WAPRICE"].replace(0, np.nan).fillna(c)
    nt  = df["NUMTRADES"].fillna(0)
    lr  = np.log((c / c.shift(1)).clip(lower=eps))
    f: dict[str, pd.Series] = {}

    # A. Momentum
    f["R1"]            = c.pct_change(1).shift(1)
    f["R5"]            = c.pct_change(5).shift(1)
    f["R21"]           = c.pct_change(21).shift(1)
    f["R63"]           = c.pct_change(63).shift(1)
    f["R126"]          = c.pct_change(126).shift(1)
    f["R12_1"]         = c.shift(21) / c.shift(252).replace(0, np.nan) - 1
    f["TSMOM"]         = np.sign(f["R63"])
    f["dist_52w_high"] = (c - h.rolling(252).max()) / h.rolling(252).max().replace(0, np.nan)
    r5 = c.pct_change(5)
    f["price_accel"]   = r5 - r5.shift(5)

    # B. Mean Reversion
    sma20    = c.rolling(20).mean()
    std20    = c.rolling(20).std().replace(0, np.nan)
    bbu      = sma20 + 2*std20;  bbl = sma20 - 2*std20
    f["Z_SMA20"]       = (c - sma20) / std20
    f["BB_pct_b"]      = (c - bbl) / (bbu - bbl).replace(0, np.nan)
    f["RSI_14"]        = _wilder_rsi(c, 14)
    h14, l14           = h.rolling(14).max(), l.rolling(14).min()
    f["Williams_R_14"] = (h14 - c) / (h14 - l14).replace(0, np.nan) * -100
    vwap5              = (wap*vol).rolling(5).sum() / vol.rolling(5).sum().replace(0, np.nan)
    f["VWAP_dev_5d"]   = (c - vwap5) / vwap5.replace(0, np.nan)

    # C. Volatility
    f["RV20"]             = lr.rolling(20).std() * np.sqrt(252)
    gk = 0.5*np.log((h/l).clip(lower=eps))**2 - (2*np.log(2)-1)*np.log((c/o).clip(lower=eps))**2
    f["GK_vol_10d"]       = np.sqrt(gk.rolling(10).mean().clip(lower=0)*252)
    pk = np.log((h/l).clip(lower=eps))**2 / (4*np.log(2))
    f["Parkinson_vol_10d"]= np.sqrt(pk.rolling(10).mean()*252)
    rs = (np.log((h/c).clip(lower=eps))*np.log((h/o).clip(lower=eps))
          + np.log((l/c).clip(lower=eps))*np.log((l/o).clip(lower=eps)))
    f["RS_vol_10d"]       = np.sqrt(rs.rolling(10).mean().clip(lower=0)*252)
    f["YZ_vol_10d"]       = _yz_volatility(o, h, l, c, n=10)
    f["BB_BandWidth"]     = (bbu - bbl) / sma20.replace(0, np.nan)
    f["ATR_pct"]          = _wilder_atr(h, l, c, 14) / c * 100
    f["vol_of_vol_60d"]   = f["RV20"].rolling(60).std()
    f["skew_20d"]         = lr.rolling(20).skew()
    f["kurt_20d"]         = lr.rolling(20).kurt()

    # D. Volume / Liquidity
    mv20, sv20            = vol.rolling(20).mean(), vol.rolling(20).std().replace(0, np.nan)
    f["V_zscore_20d"]     = (vol - mv20) / sv20
    f["Amihud_ILLIQ_20d"] = (lr.abs() / val.replace(0, np.nan)).rolling(20).median() * 1e6
    obv                   = (np.sign(c.diff()).fillna(0) * vol).cumsum()
    f["OBV_slope"]        = obv.diff(10) / vol.rolling(10).mean().replace(0, np.nan)
    mnt, snt              = nt.rolling(20).mean(), nt.rolling(20).std().replace(0, np.nan)
    f["NUMTRADES_zscore_20d"] = (nt - mnt) / snt
    f["ADV_ratio_5_60"]   = vol.rolling(5).mean() / vol.rolling(60).mean().replace(0, np.nan)
    f["turnover_rel"]     = val / val.rolling(20).mean().replace(0, np.nan)

    # F. Technical
    macd   = c.ewm(span=12,adjust=False).mean() - c.ewm(span=26,adjust=False).mean()
    sig    = macd.ewm(span=9, adjust=False).mean()
    f["MACD_hist"]       = macd - sig
    f["MACD_slope_sign"] = np.sign(f["MACD_hist"]) - np.sign(f["MACD_hist"].shift(1))
    f["Stoch_K_14"]      = ((c - l.rolling(14).min())
                             / (h.rolling(14).max()-l.rolling(14).min()).replace(0,np.nan)*100)
    f["ADX_14"]          = _adx(h, l, c, 14)
    tp   = (h+l+c)/3
    mdev = tp.rolling(20).apply(lambda x: np.abs(x-x.mean()).mean(), raw=True)
    f["CCI_20"]          = (tp - tp.rolling(20).mean()) / (0.015*mdev.replace(0, np.nan))
    f["EMA_cross_20_50"] = np.sign(c.ewm(span=20,adjust=False).mean()
                                   - c.ewm(span=50,adjust=False).mean())
    f["Donchian_pos_20"] = ((c - l.rolling(20).min())
                            / (h.rolling(20).max()-l.rolling(20).min()).replace(0, np.nan))

    # G. Cross-sectional / Beta
    ns = pd.Series(np.nan, index=c.index)
    if imoex_ret is not None:
        im = imoex_ret.reindex(c.index)
        f["Beta_IMOEX_60d"]  = _rolling_beta(lr, im, 60)
        f["RS_vs_IMOEX_20d"] = (np.exp(lr.rolling(20).sum())
                                 / np.exp(im.rolling(20).sum()).replace(0, np.nan))
        f["IVOL_60d"]        = _rolling_ivol(lr, im, 60)
    else:
        f["Beta_IMOEX_60d"] = f["RS_vs_IMOEX_20d"] = f["IVOL_60d"] = ns.copy()

    f["Beta_USDRUB_60d"] = (_rolling_beta(lr, usdrub_ret.reindex(c.index), 60)
                            if usdrub_ret is not None else ns.copy())
    f["RVI_level"]       = rvi.reindex(c.index) if rvi is not None else ns.copy()

    return pd.DataFrame(f).reset_index()


def add_calendar_features(feat_df: pd.DataFrame, divs: pd.DataFrame) -> pd.DataFrame:
    out = feat_df.copy()
    out["day_of_week"]      = out["TRADEDATE"].dt.dayofweek
    out["month"]            = out["TRADEDATE"].dt.month
    out["dte_ex_days"]      = 999
    out["post_ex_flag_10d"] = 0
    if divs.empty or "registryclosedate" not in divs.columns:
        return out
    ex  = np.array(divs["registryclosedate"].dropna().apply(_get_ex_date)
                   .sort_values().values, dtype="datetime64[D]")
    dt  = np.array(out["TRADEDATE"].values, dtype="datetime64[D]")
    dte, post = [], []
    for d in dt:
        fut = ex[ex > d]
        dte.append(int((fut.min()-d)/np.timedelta64(1,"D")) if len(fut) else 999)
        pas = ex[ex <= d]
        ds  = int((d-pas.max())/np.timedelta64(1,"D")) if len(pas) else 999
        post.append(1 if ds <= 10 else 0)
    out["dte_ex_days"]      = np.minimum(dte, 999)
    out["post_ex_flag_10d"] = post
    return out


def calculate_target(
    adj_high: pd.Series, adj_close: pd.Series, divs: pd.DataFrame
) -> pd.Series:
    future  = pd.concat([adj_high.shift(-i) for i in range(1, 6)], axis=1)
    has_all = future.notna().sum(axis=1) == 5
    target  = pd.Series(np.nan, index=adj_high.index, name="target")
    valid   = has_all & adj_close.notna() & (adj_close > 0)
    target[valid] = future[valid].max(axis=1) / adj_close[valid] - 1
    if not divs.empty and "registryclosedate" in divs.columns:
        for ex in divs["registryclosedate"].dropna().apply(_get_ex_date):
            mask = (target.index >= ex - pd.Timedelta(days=7)) & (target.index < ex)
            target[mask] = np.nan
    return target


# ═══════════════════════════════════════════════════════════════════════════════
# НОРМАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def winsorize_cs(df: pd.DataFrame, cols: list[str], lo=0.01, hi=0.99) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = df.groupby("TRADEDATE")[col].transform(
            lambda x: x.clip(x.quantile(lo), x.quantile(hi)))
    return df


def normalize_cs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    def _rz(x):
        r = x.rank(pct=True, na_option="keep")
        mu, sigma = r.mean(), r.std()
        return (r-mu)/sigma if sigma > 0 else pd.Series(0.0, index=x.index)
    for col in cols:
        df[col] = df.groupby("TRADEDATE")[col].transform(_rz)
    return df


def normalize_ts(df: pd.DataFrame, cols: list[str], window: int = 120) -> pd.DataFrame:
    df = df.copy().sort_values("TRADEDATE").reset_index(drop=True)
    for col in cols:
        rm = df[col].rolling(window, min_periods=60).mean()
        rs = df[col].rolling(window, min_periods=60).std().replace(0, np.nan)
        df[col] = (df[col] - rm) / rs
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# СИНХРОННАЯ ОБРАБОТКА ОДНОГО ТИКЕРА (CPU-BOUND)
# ═══════════════════════════════════════════════════════════════════════════════

def process_ticker(
    ticker: str, raw: pd.DataFrame, divs: pd.DataFrame,
    imoex_ret: pd.Series | None,
    usdrub_ret: pd.Series | None,
    rvi_series: pd.Series | None,
) -> pd.DataFrame | None:
    if raw.empty or len(raw) < 300:
        logger.warning("%s: %d строк, пропуск", ticker, len(raw))
        return None
    adj      = backward_adjust(raw, divs)
    feat_df  = calculate_features(adj, imoex_ret, usdrub_ret, rvi_series)
    feat_df  = add_calendar_features(feat_df, divs)
    adj_idx  = adj.set_index("TRADEDATE").sort_index()
    target   = calculate_target(adj_idx["ADJ_HIGH"], adj_idx["ADJ_CLOSE"], divs)
    tdf      = target.reset_index(); tdf.columns = ["TRADEDATE", "target"]
    out      = feat_df.merge(tdf, on="TRADEDATE", how="left")
    out.insert(1, "ticker", ticker)
    out.insert(2, "sector", TICKERS_SECTORS.get(ticker, "OTHER"))
    out      = out[out["TRADEDATE"] >= DATA_START].reset_index(drop=True)
    min_v    = int(len(FEATURE_COLS) * (1 - NAN_THRESHOLD))
    out      = out.dropna(subset=FEATURE_COLS, thresh=min_v).dropna(subset=["target"])
    if len(out) < 50:
        logger.warning("%s: после фильтрации %d строк, пропуск", ticker, len(out))
        return None
    return out.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ASYNC ПАЙПЛАЙН
# ═══════════════════════════════════════════════════════════════════════════════

async def build_all_datasets_async() -> None:
    """
    1. Параллельно: IMOEX + USDRUB + RVI через asyncio.gather
    2. Параллельно: все тикеры (history + dividends per ticker), семафор CONCURRENCY
    3. Синхронно: pandas-обработка, нормализация, сохранение в parquet
    """
    t0 = time.perf_counter()
    for sub in ["market", "sectors", "tickers"]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(CONCURRENCY)

    async with _make_session() as session:

        # ── Шаг 1: Бенчмарки — параллельно ──────────────────────────────────
        logger.info("=== [1/4] Бенчмарки параллельно (IMOEX + USDRUB + RVI) ===")
        imoex_df, usdrub_df, rvi_df = await asyncio.gather(
            fetch_index(session, sem, "IMOEX"),
            fetch_usdrub(session, sem),
            fetch_index(session, sem, "RVI"),
        )
        logger.info("  IMOEX: %d  USDRUB: %d  RVI: %d строк",
                    len(imoex_df), len(usdrub_df), len(rvi_df))

        # ── Шаг 2: Тикеры — параллельно (history + divs per ticker) ─────────
        tickers = list(TICKERS_SECTORS.keys())
        logger.info("=== [2/4] Тикеры параллельно (%d шт., CONCURRENCY=%d) ===",
                    len(tickers), CONCURRENCY)
        tasks      = [fetch_ticker_data(session, sem, t) for t in tickers]
        raw_results = await atqdm.gather(*tasks, desc="Загрузка ISS", unit="тикер")

    # ── Шаг 3: Синхронная CPU-обработка ──────────────────────────────────────
    def _log_ret(df: pd.DataFrame, col: str = "CLOSE") -> pd.Series | None:
        if df.empty or col not in df.columns:
            return None
        s = df.set_index("TRADEDATE")[col].astype(float).replace(0, np.nan)
        return np.log(s / s.shift(1))

    imoex_ret  = _log_ret(imoex_df)
    usdrub_ret = _log_ret(usdrub_df, "USDRUB")
    rvi_series = rvi_df.set_index("TRADEDATE")["CLOSE"].astype(float) if not rvi_df.empty else None

    t_io = time.perf_counter()
    logger.info("=== [3/4] Обработка тикеров (pandas) — I/O занял %.1f сек ===", t_io - t0)

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for ticker, (raw, divs) in zip(tickers, raw_results):
        try:
            frame = process_ticker(ticker, raw, divs, imoex_ret, usdrub_ret, rvi_series)
            if frame is not None:
                frames.append(frame)
        except Exception as exc:
            logger.error("Ошибка %s: %s", ticker, exc)
            failed.append(ticker)

    if failed:
        logger.warning("Тикеры с ошибками: %s", failed)
    if not frames:
        logger.error("Нет данных. Завершение.")
        return

    # ── Шаг 4: Сборка и сохранение датасетов ─────────────────────────────────
    logger.info("=== [4/4] Нормализация + сохранение ===")
    panel = pd.concat(frames, ignore_index=True).sort_values(["TRADEDATE", "ticker"])

    # Market
    pnorm   = normalize_cs(winsorize_cs(panel, NORM_COLS), NORM_COLS)
    mkt_p   = OUTPUT_DIR / "market" / "dataset_market.parquet"
    pnorm.to_parquet(mkt_p, index=False, engine="pyarrow", compression="snappy")
    logger.info("  market: %d строк | %d тикеров | %.1f MB",
                len(pnorm), pnorm["ticker"].nunique(), mkt_p.stat().st_size/1e6)

    # Sectors
    for sector in sorted(panel["sector"].unique()):
        sd = panel[panel["sector"] == sector]
        if sd["ticker"].nunique() < 2 or len(sd) < 200:
            continue
        sn = normalize_cs(winsorize_cs(sd.copy(), NORM_COLS), NORM_COLS)
        sn.to_parquet(OUTPUT_DIR/"sectors"/f"dataset_{sector}.parquet",
                      index=False, engine="pyarrow", compression="snappy")
        logger.info("  %s: %d строк", sector, len(sn))

    # Per-ticker
    for frame in frames:
        ticker = frame["ticker"].iloc[0]
        tn     = normalize_ts(frame.copy(), NORM_COLS, window=120)
        min_v  = int(len(FEATURE_COLS)*(1 - NAN_THRESHOLD))
        tn     = tn.dropna(subset=NORM_COLS, thresh=min_v).dropna(subset=["target"])
        if len(tn) < 50:
            continue
        tn.to_parquet(OUTPUT_DIR/"tickers"/f"dataset_{ticker}.parquet",
                      index=False, engine="pyarrow", compression="snappy")

    t_total = time.perf_counter()
    logger.info("=" * 65)
    logger.info("ГОТОВО: %.1f сек total (I/O %.1f сек | CPU %.1f сек)",
                t_total - t0, t_io - t0, t_total - t_io)
    logger.info("Market: %d строк | %d тикеров | %d фичей",
                len(pnorm), pnorm["ticker"].nunique(), len(FEATURE_COLS))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Jupyter: используйте  `await build_all_datasets_async()`
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(build_all_datasets_async())