# %% [markdown]
# # Обучение ML-моделей: прогноз max-5d роста акций MOEX
#
# **Целевая переменная:** `target = max(ADJ_HIGH[t+1..t+5]) / ADJ_CLOSE[t] - 1`
#
# **Модели:** Ridge · ElasticNet · HuberRegressor · LightGBM
#
# **CV:** PurgedKFold + WalkForward (оба, с comparison)
#
# **Метрики:** MSE · R² · MAE · Spearman IC · ICIR · Directional Accuracy
#
# **Запуск как notebook:**
# ```
# pip install jupytext
# jupytext --to notebook train_models.py
# jupyter notebook train_models.ipynb
# ```
# Или открыть `train_models.py` напрямую в VS Code (Python Interactive).

# %% [markdown]
# ## 1. Импорты и конфигурация

# %%
import warnings
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Пути ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("datasets")
RESULTS_DIR = Path("results")
MODELS_DIR  = Path("models")
FIGS_DIR    = RESULTS_DIR / "figures"

for d in [RESULTS_DIR, MODELS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Признаки ────────────────────────────────────────────────────────────────
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

# ─── Гиперпараметры обучения ─────────────────────────────────────────────────
# n_trials: количество Optuna-испытаний на каждую модель × датасет
# timeout_sec: макс. время (сек) на оптимизацию одной модели
OPTUNA_CFG = {
    "market":  {"n_trials": 40, "timeout_sec": 600},
    "sector":  {"n_trials": 30, "timeout_sec": 300},
    "ticker":  {"n_trials": 20, "timeout_sec": 120},
}

PURGED_SPLITS   = 5      # складки PurgedKFold
WF_SPLITS       = 5      # складки WalkForward
EMBARGO_DAYS    = 5      # дней эмбарго между train и test
PURGE_DAYS      = 5      # дней purge (горизонт таргета)
MIN_TRAIN_ROWS  = 200    # минимум строк в обучающей части

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Защита данных ─────────────────────────────────────────────────────────────
# Предупреждение при обнаружении таргета за пределами физически разумного диапазона.
# Если build_datasets.py отработал корректно, значений за этими границами быть не должно.
TARGET_ANOMALY_THRESHOLD = 1.00   # >100% за 5 дней = аномалия данных
# Верхняя граница оси для графика распределения таргета (99.9-й перцентиль, но не более этого)
TARGET_VIZ_MAX_PCT = 25.0         # % — разумный потолок для визуализации

print(f"Feature count : {len(FEATURE_COLS)}")
print(f"Data dir      : {DATA_DIR.resolve()}")

# %% [markdown]
# ## 2. Загрузка данных

# %%
def load_parquet(path: Path) -> pd.DataFrame | None:
    """Безопасная загрузка parquet с проверкой минимального размера и аномалий таргета."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "TRADEDATE" in df.columns:
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    if len(df) < 50:
        return None

    # Проверяем наличие аномальных значений таргета (признак необработанных сплитов)
    if "target" in df.columns:
        t = df["target"].dropna()
        n_anomaly = int((t.abs() > TARGET_ANOMALY_THRESHOLD).sum())
        if n_anomaly:
            logger.warning(
                "load_parquet [%s]: обнаружено %d строк с |target| > %.0f%% "
                "(max=%.1f%%). Пересоберите датасет через build_datasets.py.",
                path.name, n_anomaly, TARGET_ANOMALY_THRESHOLD * 100,
                float(t.max()) * 100,
            )

    return df


# Market
df_market = load_parquet(DATA_DIR / "market" / "dataset_market.parquet")

# Sectors
sector_paths = sorted((DATA_DIR / "sectors").glob("*.parquet"))
sector_dfs: dict[str, pd.DataFrame] = {}
for p in sector_paths:
    name = p.stem.replace("dataset_", "")
    df_s = load_parquet(p)
    if df_s is not None:
        sector_dfs[name] = df_s

# Tickers
ticker_paths = sorted((DATA_DIR / "tickers").glob("*.parquet"))
ticker_dfs: dict[str, pd.DataFrame] = {}
for p in ticker_paths:
    name = p.stem.replace("dataset_", "")
    df_t = load_parquet(p)
    if df_t is not None:
        ticker_dfs[name] = df_t

print(f"Market  : {'✓ ' + str(len(df_market)) + ' rows' if df_market is not None else '✗ не найден'}")
print(f"Sectors : {len(sector_dfs)} файлов  → {list(sector_dfs.keys())}")
print(f"Tickers : {len(ticker_dfs)} файлов")

# %% [markdown]
# ## 3. EDA — базовая статистика и распределение таргета

# %%
def eda_summary(df: pd.DataFrame, name: str = "dataset") -> None:
    """Быстрый EDA: shape, target dist, NaN summary."""
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")
    print(f"  Строк × колонок: {df.shape[0]:,} × {df.shape[1]}")
    if "ticker" in df.columns:
        print(f"  Тикеров        : {df['ticker'].nunique()}")
    if "TRADEDATE" in df.columns:
        print(f"  Период         : {df['TRADEDATE'].min().date()} → {df['TRADEDATE'].max().date()}")
    if "target" in df.columns:
        t = df["target"].dropna()
        n_anom = int((t.abs() > TARGET_ANOMALY_THRESHOLD).sum())
        anom_flag = f"  ⚠  АНОМАЛИИ: {n_anom} строк с |target|>100%" if n_anom else ""
        print(f"  Target (%)     : mean={t.mean()*100:.2f}  "
              f"std={t.std()*100:.2f}  "
              f"median={t.median()*100:.2f}  "
              f"p5={t.quantile(.05)*100:.2f}  "
              f"p95={t.quantile(.95)*100:.2f}  "
              f"max={t.max()*100:.1f}")
        if anom_flag:
            print(anom_flag)

    feat_present = [c for c in FEATURE_COLS if c in df.columns]
    nan_frac = df[feat_present].isna().mean().sort_values(ascending=False)
    heavy_nan = nan_frac[nan_frac > 0.10]
    if not heavy_nan.empty:
        print(f"\n  Признаки с >10% NaN:")
        for col, frac in heavy_nan.items():
            print(f"    {col:<28} {frac*100:.1f}%")


if df_market is not None:
    eda_summary(df_market, "Market dataset")

# %%
# ─── Визуализация распределения таргета ──────────────────────────────────────
if df_market is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    t_raw = df_market["target"].dropna() * 100   # в процентах

    # Клипинг для визуализации: p99.9 или жёсткий потолок TARGET_VIZ_MAX_PCT
    viz_clip = min(float(t_raw.quantile(0.999)), TARGET_VIZ_MAX_PCT)
    t_vals   = t_raw.clip(upper=viz_clip)
    n_clipped = int((t_raw > viz_clip).sum())

    axes[0].hist(t_vals, bins=80, color="#2563EB", edgecolor="none", alpha=0.85)
    axes[0].axvline(0, color="red", lw=1.5, linestyle="--", label="0%")
    axes[0].set_xlabel("max 5d return (%)", fontsize=11)
    axes[0].set_ylabel("Частота", fontsize=11)
    axes[0].set_title("Распределение таргета", fontsize=12, fontweight="bold")
    if n_clipped:
        axes[0].annotate(
            f"обрезано {n_clipped} точек\n(>{viz_clip:.0f}%)",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=8.5, color="#DC2626",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DC2626", alpha=0.8),
        )
    axes[0].legend()

    # По годам — используем медиану вместо mean, чтобы выбросы не искажали график
    yearly = (df_market.assign(year=df_market["TRADEDATE"].dt.year)
                        .groupby("year")["target"]
                        .agg(median="median", mean="mean") * 100)
    x_pos = range(len(yearly))
    axes[1].bar(x_pos, yearly["mean"],   alpha=0.55, color="#2563EB", label="mean")
    axes[1].bar(x_pos, yearly["median"], alpha=0.85, color="#1D4ED8", label="median")
    axes[1].set_xticks(list(x_pos))
    axes[1].set_xticklabels(yearly.index.astype(str))
    axes[1].set_title("Таргет по годам (mean + median)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Год")
    axes[1].set_ylabel("target (%)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGS_DIR / "target_distribution.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Сохранено:", FIGS_DIR / "target_distribution.png")

# %% [markdown]
# ## 4. IC-анализ признаков

# %%
def compute_feature_ic(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Вычисляет Spearman IC между каждым признаком и таргетом.
    Для панельных данных — IC считается per-date, затем усредняется (mean IC и ICIR).
    """
    rows = []
    present = [c for c in feature_cols if c in df.columns and "target" in df.columns]

    if "TRADEDATE" in df.columns:
        # Панель: IC per date
        dates = df["TRADEDATE"].unique()
        ic_by_date: dict[str, list] = {c: [] for c in present}
        for date in dates:
            sub = df[df["TRADEDATE"] == date][present + ["target"]].dropna(subset=["target"])
            if len(sub) < 5:
                continue
            for col in present:
                vals = sub[[col, "target"]].dropna()
                if len(vals) < 5:
                    continue
                ic, _ = spearmanr(vals[col], vals["target"])
                if not np.isnan(ic):
                    ic_by_date[col].append(ic)
        for col in present:
            ics = ic_by_date[col]
            if ics:
                mean_ic = np.mean(ics)
                std_ic  = np.std(ics) + 1e-8
                rows.append({"feature": col, "mean_IC": mean_ic,
                             "std_IC": std_ic, "ICIR": mean_ic / std_ic,
                             "n_dates": len(ics)})
    else:
        # Один тикер: IC по всему ряду
        for col in present:
            vals = df[[col, "target"]].dropna()
            if len(vals) < 10:
                continue
            ic, _ = spearmanr(vals[col], vals["target"])
            rows.append({"feature": col, "mean_IC": ic, "std_IC": np.nan,
                         "ICIR": np.nan, "n_dates": len(vals)})

    return (pd.DataFrame(rows)
              .sort_values("mean_IC", key=abs, ascending=False)
              .reset_index(drop=True))


# %%
if df_market is not None:
    ic_df = compute_feature_ic(df_market, FEATURE_COLS)
    print(ic_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ["#2563EB" if v > 0 else "#DC2626" for v in ic_df["mean_IC"]]
    ax.barh(ic_df["feature"], ic_df["mean_IC"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("mean Spearman IC", fontsize=11)
    ax.set_title("IC признаков (market dataset, cross-sectional per day)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "feature_ic_market.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Сохранено:", FIGS_DIR / "feature_ic_market.png")
    ic_df.to_csv(RESULTS_DIR / "feature_ic.csv", index=False)

# %% [markdown]
# ## 5. CV-стратегии

# %%
class PurgedKFold:
    """
    Purged Walk-Forward K-Fold CV (по Lopez de Prado, AFML гл. 7).

    Ключевые идеи:
    - K non-overlapping тестовых фолдов, расположенных последовательно во времени
    - Обучение на ВСЕХ предшествующих фолдах (expanding window)
    - Purge: из обучения удаляются строки, чей target-горизонт перекрывается с тестом
    - Embargo: после теста удаляется буфер `embargo_days` дней из следующей выборки

    Параметры
    ----------
    n_splits     : число тестовых фолдов
    purge_days   : горизонт таргета (дней) для удаления граничных обучающих строк
    embargo_days : буфер (дней) между концом train и началом test
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 5, embargo_days: int = 5):
        self.n_splits    = n_splits
        self.purge_days  = purge_days
        self.embargo_days = embargo_days

    def split(self, X: pd.DataFrame, y=None, dates: pd.Series = None):
        """
        Возвращает (train_idx, test_idx) для каждого фолда.
        dates: pd.Series с датами, выровненная по X.index
        """
        assert dates is not None, "Передайте dates=pd.Series дат"
        dates = pd.to_datetime(dates).reset_index(drop=True)
        unique_dates = np.sort(dates.unique())
        n = len(unique_dates)
        fold_size = n // (self.n_splits + 1)

        for k in range(1, self.n_splits + 1):
            t_start = unique_dates[k * fold_size]
            t_end   = unique_dates[min((k + 1) * fold_size - 1, n - 1)]

            # Purge + embargo: отступаем назад от t_start
            cutoff = t_start - pd.Timedelta(days=self.purge_days + self.embargo_days)

            train_idx = np.where(dates < cutoff)[0]
            test_idx  = np.where((dates >= t_start) & (dates <= t_end))[0]

            if len(train_idx) >= MIN_TRAIN_ROWS and len(test_idx) >= 10:
                yield train_idx, test_idx


class WalkForwardCV:
    """
    Walk-Forward Expanding Window CV.

    На каждом шаге:
    - Train: все данные до t_start - embargo_days (expanding window)
    - Test : один фолд начиная с t_start

    Параметры
    ----------
    n_splits     : число шагов
    min_train    : минимальный размер обучающей выборки (строк)
    embargo_days : буфер между концом train и началом test
    """

    def __init__(self, n_splits: int = 5, min_train: int = MIN_TRAIN_ROWS, embargo_days: int = 5):
        self.n_splits    = n_splits
        self.min_train   = min_train
        self.embargo_days = embargo_days

    def split(self, X: pd.DataFrame, y=None, dates: pd.Series = None):
        assert dates is not None, "Передайте dates=pd.Series дат"
        dates = pd.to_datetime(dates).reset_index(drop=True)
        unique_dates = np.sort(dates.unique())
        n = len(unique_dates)

        # Первый допустимый тест — после min_train строк
        first_test_date_idx = np.searchsorted(
            unique_dates,
            dates.iloc[self.min_train - 1] if len(dates) > self.min_train else unique_dates[-1]
        )
        remaining = n - first_test_date_idx
        if remaining < self.n_splits:
            return

        fold_size = remaining // self.n_splits

        for k in range(self.n_splits):
            t_start_idx = first_test_date_idx + k * fold_size
            t_end_idx   = min(first_test_date_idx + (k + 1) * fold_size - 1, n - 1)
            if t_start_idx >= n:
                break

            t_start = unique_dates[t_start_idx]
            t_end   = unique_dates[t_end_idx]
            embargo_date = t_start - pd.Timedelta(days=self.embargo_days)

            train_idx = np.where(dates < embargo_date)[0]
            test_idx  = np.where((dates >= t_start) & (dates <= t_end))[0]

            if len(train_idx) >= self.min_train and len(test_idx) >= 10:
                yield train_idx, test_idx


# Проверяем сплиты на market dataset
if df_market is not None:
    dates_mkt = df_market["TRADEDATE"]
    pkf = PurgedKFold(n_splits=PURGED_SPLITS)
    wf  = WalkForwardCV(n_splits=WF_SPLITS)

    print("=== PurgedKFold splits ===")
    for i, (tr, te) in enumerate(pkf.split(df_market, dates=dates_mkt)):
        d_tr = dates_mkt.iloc[tr]
        d_te = dates_mkt.iloc[te]
        print(f"  Fold {i+1}: train {d_tr.min().date()}→{d_tr.max().date()} "
              f"({len(tr):,} rows) | test {d_te.min().date()}→{d_te.max().date()} ({len(te):,} rows)")

    print("\n=== WalkForward splits ===")
    for i, (tr, te) in enumerate(wf.split(df_market, dates=dates_mkt)):
        d_tr = dates_mkt.iloc[tr]
        d_te = dates_mkt.iloc[te]
        print(f"  Step {i+1}: train→{d_tr.max().date()} ({len(tr):,} rows) "
              f"| test {d_te.min().date()}→{d_te.max().date()} ({len(te):,} rows)")

# %% [markdown]
# ## 6. Метрики

# %%
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Все ключевые метрики для регрессии на финансовый таргет.

    Возвращает dict:
        MSE, MAE, R2, IC (Spearman), Dir_Acc
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 5:
        return {"MSE": np.nan, "MAE": np.nan, "R2": np.nan, "IC": np.nan, "Dir_Acc": np.nan}

    mse = float(np.mean((yt - yp) ** 2))
    mae = float(np.mean(np.abs(yt - yp)))

    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2  = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    ic, _    = spearmanr(yt, yp)
    dir_acc  = float(np.mean(np.sign(yt) == np.sign(yp)))

    return {"MSE": mse, "MAE": mae, "R2": r2, "IC": float(ic), "Dir_Acc": dir_acc}


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """
    Агрегирует метрики по фолдам: mean, std + ICIR.
    """
    if not fold_metrics:
        return {}
    df_m = pd.DataFrame(fold_metrics)
    result = {}
    for col in df_m.columns:
        result[f"{col}_mean"] = df_m[col].mean()
        result[f"{col}_std"]  = df_m[col].std()
    ic_vals = df_m["IC"].dropna()
    result["ICIR"] = (ic_vals.mean() / (ic_vals.std() + 1e-8)) if len(ic_vals) > 0 else np.nan
    return result

# %% [markdown]
# ## 7. Препроцессинг и модели

# %%
def build_X_y(
    df: pd.DataFrame,
    feat_cols: list[str] | None = None,
    clip_target: bool = True,
) -> tuple:
    """
    Возвращает (X, y, dates) из DataFrame.
    Дропает строки без таргета.

    clip_target: если True, обрезает таргет по тем же границам что build_datasets.py
                 (защита на случай старых parquet-файлов без клипинга).
    """
    if feat_cols is None:
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    sub = df.dropna(subset=["target"]).reset_index(drop=True)

    if clip_target:
        # Повторяем клипинг из build_datasets для совместимости со старыми файлами
        clip_lo, clip_hi = -0.50, TARGET_ANOMALY_THRESHOLD
        n_clipped = int(((sub["target"] < clip_lo) | (sub["target"] > clip_hi)).sum())
        if n_clipped:
            logger.warning(
                "build_X_y: обрезано %d аномальных значений таргета. "
                "Рекомендуется пересобрать датасеты через build_datasets.py.",
                n_clipped,
            )
            sub = sub.copy()
            sub["target"] = sub["target"].clip(lower=clip_lo, upper=clip_hi)

    X     = sub[feat_cols].copy()
    y     = sub["target"].copy()
    dates = sub["TRADEDATE"] if "TRADEDATE" in sub.columns else pd.Series(range(len(sub)))
    return X, y, dates


def make_pipeline(model_name: str, hparams: dict) -> Pipeline:
    """
    Строит sklearn Pipeline: Imputer → Scaler → Model.
    LightGBM не требует стандартизации, но мы оставляем для единообразия.
    """
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if model_name == "Ridge":
        model = Ridge(alpha=hparams.get("alpha", 1.0), random_state=RANDOM_SEED)
    elif model_name == "ElasticNet":
        model = ElasticNet(
            alpha=hparams.get("alpha", 0.1),
            l1_ratio=hparams.get("l1_ratio", 0.5),
            max_iter=10_000,
            random_state=RANDOM_SEED,
        )
    elif model_name == "Huber":
        model = HuberRegressor(
            epsilon=hparams.get("epsilon", 1.35),
            alpha=hparams.get("alpha", 0.01),
            max_iter=500,
        )
    elif model_name == "LightGBM":
        model = lgb.LGBMRegressor(
            n_estimators=hparams.get("n_estimators", 200),
            max_depth=hparams.get("max_depth", 5),
            learning_rate=hparams.get("learning_rate", 0.05),
            num_leaves=hparams.get("num_leaves", 31),
            min_child_samples=hparams.get("min_child_samples", 30),
            subsample=hparams.get("subsample", 0.8),
            colsample_bytree=hparams.get("colsample_bytree", 0.8),
            reg_alpha=hparams.get("reg_alpha", 0.0),
            reg_lambda=hparams.get("reg_lambda", 1.0),
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    return Pipeline([("imputer", imp), ("scaler", scaler), ("model", model)])


def _suggest_hparams(trial, model_name: str) -> dict:
    """Optuna: пространство гиперпараметров для каждой модели."""
    if model_name == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e4, log=True)}

    elif model_name == "ElasticNet":
        return {
            "alpha":    trial.suggest_float("alpha",    1e-4, 1e3, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
        }
    elif model_name == "Huber":
        return {
            "epsilon": trial.suggest_float("epsilon", 1.0, 5.0),
            "alpha":   trial.suggest_float("alpha",   1e-4, 100.0, log=True),
        }
    elif model_name == "LightGBM":
        return {
            "n_estimators":     trial.suggest_int("n_estimators",     50, 500),
            "max_depth":        trial.suggest_int("max_depth",         3,   8),
            "learning_rate":    trial.suggest_float("learning_rate",   0.01, 0.3, log=True),
            "num_leaves":       trial.suggest_int("num_leaves",        16, 127),
            "min_child_samples":trial.suggest_int("min_child_samples", 15, 100),
            "subsample":        trial.suggest_float("subsample",       0.5,  1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",0.5,  1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha",       0.0, 10.0),
            "reg_lambda":       trial.suggest_float("reg_lambda",      0.0, 10.0),
        }
    raise ValueError(model_name)

# %% [markdown]
# ## 8. Тренировочный пайплайн

# %%
def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    cv_strategy,
    model_name: str,
    hparams: dict,
) -> tuple[list[dict], list[float]]:
    """
    Запускает CV и возвращает (fold_metrics_list, fold_ic_list).
    """
    fold_metrics: list[dict] = []
    fold_ic: list[float] = []

    for train_idx, test_idx in cv_strategy.split(X, dates=dates):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipe = make_pipeline(model_name, hparams)
        try:
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
        except Exception:
            continue

        m = compute_metrics(y_te.values, y_pred)
        fold_metrics.append(m)
        if not np.isnan(m["IC"]):
            fold_ic.append(m["IC"])

    return fold_metrics, fold_ic


def optimize_model(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    model_name: str,
    cv_strategy,
    n_trials: int,
    timeout_sec: int,
) -> tuple[dict, float]:
    """
    Optuna-оптимизация гиперпараметров: максимизация mean IC.
    Возвращает (best_hparams, best_ic).
    """
    def objective(trial):
        hparams = _suggest_hparams(trial, model_name)
        _, fold_ic = run_cv(X, y, dates, cv_strategy, model_name, hparams)
        return np.mean(fold_ic) if fold_ic else -1.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)

    best_params = study.best_params
    best_ic     = study.best_value
    return best_params, best_ic


def train_and_evaluate(
    df: pd.DataFrame,
    dataset_name: str,
    dataset_type: str = "market",   # "market" | "sector" | "ticker"
    feat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Полный цикл обучения для одного датасета:
      1. Optuna → best hparams (PurgedKFold)
      2. Переобучение на лучших hparams → финальная оценка (WalkForward)
      3. Сохраняем лучшую модель в models/

    Возвращает DataFrame с результатами по всем моделям.
    """
    cfg = OPTUNA_CFG.get(dataset_type, OPTUNA_CFG["market"])
    X, y, dates = build_X_y(df, feat_cols)
    feat_cols_actual = X.columns.tolist()

    if len(X) < MIN_TRAIN_ROWS * 2:
        logger.warning("%s: слишком мало данных (%d), пропуск", dataset_name, len(X))
        return pd.DataFrame()

    n_pkf_splits = max(2, PURGED_SPLITS if len(X) >= 500 else 3)
    n_wf_splits  = max(2, WF_SPLITS     if len(X) >= 500 else 3)
    pkf = PurgedKFold(n_splits=n_pkf_splits)
    wf  = WalkForwardCV(n_splits=n_wf_splits)

    model_names = ["Ridge", "ElasticNet", "Huber", "LightGBM"]
    all_results: list[dict] = []

    for mname in model_names:
        logger.info("  [%s] %s — Optuna (%d trials, timeout=%ds)...",
                    dataset_name, mname, cfg["n_trials"], cfg["timeout_sec"])

        # ── 1. Optuna (PurgedKFold) ──────────────────────────────────────────
        best_hp, best_ic_pkf = optimize_model(
            X, y, dates, mname, pkf, cfg["n_trials"], cfg["timeout_sec"]
        )

        # ── 2. WalkForward evaluation с лучшими hparams ─────────────────────
        fold_metrics_wf, fold_ic_wf = run_cv(X, y, dates, wf, mname, best_hp)
        agg_wf  = aggregate_fold_metrics(fold_metrics_wf)

        # ── 3. PurgedKFold evaluation с лучшими hparams (для сравнения) ─────
        fold_metrics_pkf, fold_ic_pkf = run_cv(X, y, dates, pkf, mname, best_hp)
        agg_pkf = aggregate_fold_metrics(fold_metrics_pkf)

        # ── 4. Сохраняем финальную модель (обучаем на ВСЕХ данных) ──────────
        final_pipe = make_pipeline(mname, best_hp)
        final_pipe.fit(X, y)
        safe_name = dataset_name.replace("/", "_").replace(" ", "_")
        model_path = MODELS_DIR / f"{safe_name}_{mname}.pkl"
        joblib.dump({"pipeline": final_pipe, "features": feat_cols_actual,
                     "hparams": best_hp, "dataset": dataset_name}, model_path)

        row = {
            "dataset":      dataset_name,
            "model":        mname,
            "n_rows":       len(X),
            "best_IC_optuna": best_ic_pkf,
            **{f"pkf_{k}": v for k, v in agg_pkf.items()},
            **{f"wf_{k}":  v for k, v in agg_wf.items()},
            "model_path":   str(model_path),
        }
        all_results.append(row)

        # Краткий лог
        ic_wf   = agg_wf.get("IC_mean",   np.nan)
        icir_wf = agg_wf.get("ICIR",      np.nan)
        r2_wf   = agg_wf.get("R2_mean",   np.nan)
        da_wf   = agg_wf.get("Dir_Acc_mean", np.nan)
        logger.info(
            "    WF → IC=%.4f  ICIR=%.3f  R²=%.4f  DirAcc=%.3f  [best_hp=%s]",
            ic_wf, icir_wf, r2_wf, da_wf,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in best_hp.items()},
        )

    return pd.DataFrame(all_results)

# %% [markdown]
# ## 9. Запуск обучения

# %%
# ─── Market dataset ──────────────────────────────────────────────────────────
results_market = pd.DataFrame()

if df_market is not None:
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  MARKET DATASET  (%d строк)                         ║", len(df_market))
    logger.info("╚══════════════════════════════════════════════════════╝")
    results_market = train_and_evaluate(df_market, "market", dataset_type="market")
    print("\n── Market results ──────────────────────────────────────────")
    key_cols = ["model", "n_rows", "best_IC_optuna",
                "wf_IC_mean", "wf_ICIR", "wf_R2_mean", "wf_Dir_Acc_mean",
                "pkf_IC_mean", "pkf_ICIR"]
    key_cols = [c for c in key_cols if c in results_market.columns]
    print(results_market[key_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# %%
# ─── Sector datasets ─────────────────────────────────────────────────────────
results_sectors_list = []

for sector_name, df_sec in sector_dfs.items():
    logger.info("── SECTOR: %s (%d строк) ──", sector_name, len(df_sec))
    res = train_and_evaluate(df_sec, f"sector_{sector_name}", dataset_type="sector")
    if not res.empty:
        results_sectors_list.append(res)

results_sectors = pd.concat(results_sectors_list, ignore_index=True) if results_sectors_list else pd.DataFrame()

if not results_sectors.empty:
    key_cols = ["dataset", "model", "wf_IC_mean", "wf_ICIR", "wf_R2_mean", "wf_Dir_Acc_mean"]
    key_cols = [c for c in key_cols if c in results_sectors.columns]
    print("\n── Sector results (summary) ────────────────────────────────")
    print(results_sectors[key_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# %%
# ─── Per-ticker datasets ─────────────────────────────────────────────────────
results_tickers_list = []

# Ограничиваем количество одновременно обучаемых тикеров (можно убрать лимит)
MAX_TICKERS = len(ticker_dfs)   # обучаем на всех; поставьте, например, 10 для теста

ticker_items = list(ticker_dfs.items())[:MAX_TICKERS]

for ticker_name, df_tick in ticker_items:
    logger.info("── TICKER: %s (%d строк) ──", ticker_name, len(df_tick))
    res = train_and_evaluate(df_tick, f"ticker_{ticker_name}", dataset_type="ticker")
    if not res.empty:
        results_tickers_list.append(res)

results_tickers = pd.concat(results_tickers_list, ignore_index=True) if results_tickers_list else pd.DataFrame()

if not results_tickers.empty:
    summary = (results_tickers.groupby("model")[["wf_IC_mean", "wf_ICIR", "wf_Dir_Acc_mean"]]
               .agg(["mean", "std"]).round(4))
    print("\n── Per-ticker results (aggregated by model) ────────────────")
    print(summary.to_string())

# %% [markdown]
# ## 10. Сохранение результатов

# %%
# Объединяем все результаты
all_results_frames = [
    f for f in [results_market, results_sectors, results_tickers]
    if not f.empty
]
if all_results_frames:
    results_all = pd.concat(all_results_frames, ignore_index=True)
    out_path = RESULTS_DIR / "training_results.csv"
    results_all.to_csv(out_path, index=False)
    logger.info("Результаты сохранены: %s", out_path)
    print(f"\nВсего строк в results: {len(results_all)}")
    print(results_all[["dataset", "model", "wf_IC_mean", "wf_ICIR"]].to_string(index=False))

# %% [markdown]
# ## 11. Визуализация результатов

# %%
def plot_cv_ic_comparison(results_df: pd.DataFrame, title: str, save_path: Path) -> None:
    """Сравнение IC по моделям для WalkForward и PurgedKFold."""
    if results_df.empty:
        return
    wf_cols  = [c for c in results_df.columns if c == "wf_IC_mean"]
    pkf_cols = [c for c in results_df.columns if c == "pkf_IC_mean"]
    if not wf_cols or not pkf_cols:
        return

    sub = results_df[["model", "dataset", "wf_IC_mean", "pkf_IC_mean",
                       "wf_ICIR", "pkf_ICIR"]].copy()

    # По моделям (если несколько датасетов — усредняем)
    grp = sub.groupby("model")[["wf_IC_mean", "pkf_IC_mean"]].mean().reset_index()

    x = np.arange(len(grp))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, grp["wf_IC_mean"],  w, label="WalkForward",  color="#2563EB", alpha=0.85)
    ax.bar(x + w/2, grp["pkf_IC_mean"], w, label="PurgedKFold",  color="#16A34A", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(grp["model"])
    ax.set_ylabel("mean Spearman IC")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print("Сохранено:", save_path)


if not results_market.empty:
    plot_cv_ic_comparison(
        results_market,
        "IC по моделям — Market dataset (WalkForward vs PurgedKFold)",
        FIGS_DIR / "ic_comparison_market.png",
    )

if not results_sectors.empty:
    # Heatmap IC по секторам × моделям
    pivot = (results_sectors
             .pivot_table(index="dataset", columns="model", values="wf_IC_mean")
             .round(4))
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5 + 2)))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("IC (WalkForward) по секторам × моделям", fontweight="bold")
    ax.set_xlabel("Модель")
    ax.set_ylabel("Датасет")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "ic_heatmap_sectors.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Сохранено:", FIGS_DIR / "ic_heatmap_sectors.png")

# %%
# ─── IC тикеров: boxplot по моделям ─────────────────────────────────────────
if not results_tickers.empty and "wf_IC_mean" in results_tickers.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot IC по моделям
    models_present = results_tickers["model"].unique()
    ic_by_model = [results_tickers[results_tickers["model"] == m]["wf_IC_mean"].dropna().values
                   for m in models_present]
    axes[0].boxplot(ic_by_model, labels=models_present, patch_artist=True,
                    boxprops=dict(facecolor="#DBEAFE"), medianprops=dict(color="#2563EB", lw=2))
    axes[0].axhline(0, color="red", lw=1, linestyle="--")
    axes[0].set_title("Распределение IC по тикерам", fontweight="bold")
    axes[0].set_ylabel("WalkForward IC")

    # ICIR по моделям
    icir_by_model = [results_tickers[results_tickers["model"] == m]["wf_ICIR"].dropna().values
                     for m in models_present]
    axes[1].boxplot(icir_by_model, labels=models_present, patch_artist=True,
                    boxprops=dict(facecolor="#DCFCE7"), medianprops=dict(color="#16A34A", lw=2))
    axes[1].axhline(0, color="red", lw=1, linestyle="--")
    axes[1].set_title("Распределение ICIR по тикерам", fontweight="bold")
    axes[1].set_ylabel("ICIR")

    plt.suptitle("Per-ticker модели — агрегация по тикерам", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "ic_boxplot_tickers.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Сохранено:", FIGS_DIR / "ic_boxplot_tickers.png")

# %% [markdown]
# ## 12. Feature Importance

# %%
def extract_feature_importance(model_path_pattern: str) -> pd.DataFrame:
    """
    Извлекает веса/важности признаков из сохранённых моделей.
    Для Ridge/ElasticNet/Huber — |coef|.
    Для LightGBM — feature_importances_ (gain).
    """
    rows = []
    for p in sorted(MODELS_DIR.glob(model_path_pattern)):
        loaded = joblib.load(p)
        pipe     = loaded["pipeline"]
        features = loaded["features"]
        model    = pipe.named_steps["model"]
        name     = p.stem

        try:
            if hasattr(model, "coef_"):
                importance = np.abs(model.coef_)
            elif hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            else:
                continue

            for feat, imp in zip(features, importance):
                rows.append({"model_file": name, "feature": feat, "importance": float(imp)})
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()
    df_imp = pd.DataFrame(rows)
    return (df_imp.groupby("feature")["importance"]
                  .mean()
                  .sort_values(ascending=False)
                  .reset_index()
                  .rename(columns={"importance": "mean_importance"}))


# Feature importance для market моделей
imp_df = extract_feature_importance("market_*.pkl")

if not imp_df.empty:
    top30 = imp_df.head(30)
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(top30["feature"], top30["mean_importance"], color="#2563EB", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |coef| / gain (нормализовано)", fontsize=11)
    ax.set_title("Top-30 признаков по важности\n(market dataset, среднее по моделям)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "feature_importance_market.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Сохранено:", FIGS_DIR / "feature_importance_market.png")
    imp_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

# %% [markdown]
# ## 13. Итоговая сводка

# %%
print("\n" + "═" * 65)
print("  ИТОГОВАЯ СВОДКА")
print("═" * 65)

if not results_market.empty:
    best_mkt = results_market.loc[results_market["wf_IC_mean"].idxmax()]
    print(f"\n  ► Лучшая модель на market dataset:")
    print(f"    Модель   : {best_mkt['model']}")
    print(f"    WF IC    : {best_mkt.get('wf_IC_mean', float('nan')):.4f}")
    print(f"    WF ICIR  : {best_mkt.get('wf_ICIR', float('nan')):.3f}")
    print(f"    WF R²    : {best_mkt.get('wf_R2_mean', float('nan')):.4f}")
    print(f"    Dir.Acc  : {best_mkt.get('wf_Dir_Acc_mean', float('nan')):.3f}")
    print(f"    Модель   : {best_mkt.get('model_path', 'N/A')}")

print(f"\n  Сохранённые модели   : {len(list(MODELS_DIR.glob('*.pkl')))}")
print(f"  Таблица результатов  : {RESULTS_DIR / 'training_results.csv'}")
print(f"  Важность признаков   : {RESULTS_DIR / 'feature_importance.csv'}")
print(f"  Графики              : {FIGS_DIR}/")
print("\n  ⚠  Типичные IC для краткосрочных квант-моделей: 0.02–0.05 out-of-sample")
print("     IC < 0.01 в течение 6 месяцев → рассмотрите gradient boosting / другие фичи")
print("     Отрицательный β при volatility-блоке → проверьте ex-div маскировку в таргете")
