"""Causal, scale-invariant EMA controls for daily price forecasts."""

import numpy as np
import pandas as pd


def ema_features(close: pd.Series, spans=(20, 60), min_history=120) -> pd.DataFrame:
    """Reset EMA after missing sessions; never fill missing execution prices."""
    if min_history < 21 or any(span < 2 for span in spans):
        raise ValueError("min_history >= 21 and spans >= 2 are required")
    close = pd.to_numeric(close, errors="coerce").astype(float)
    logp = np.log(close.where(np.isfinite(close) & (close > 0)))
    names = [f"ema{span}_{name}" for span in spans
             for name in ("dev", "slope5", "residual_scale")]
    result = pd.DataFrame(np.nan, index=close.index, columns=names)
    finite = np.isfinite(logp.to_numpy())
    starts = np.flatnonzero(finite & ~np.r_[False, finite[:-1]])
    ends = np.flatnonzero(finite & ~np.r_[finite[1:], False]) + 1
    for start, end in zip(starts, ends):
        if end - start < min_history:
            continue
        series = logp.iloc[start:end]
        sigma = series.diff().rolling(20, min_periods=20).std(ddof=1)
        sigma = sigma.where(sigma > 1e-8)
        for span in spans:
            smooth = series.ewm(span=span, adjust=False).mean()
            residual = series - smooth
            block = pd.DataFrame({
                f"ema{span}_dev": residual / sigma,
                f"ema{span}_slope5": smooth.diff(5) / (5 * sigma),
                f"ema{span}_residual_scale": residual.rolling(
                    min_history, min_periods=min_history).std(ddof=1) / sigma,
            })
            block.iloc[:min_history - 1] = np.nan
            result.loc[block.index, block.columns] = block
    return result
