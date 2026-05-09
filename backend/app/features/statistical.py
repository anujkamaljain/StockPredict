"""
Statistical feature computation.
Rolling statistics, distribution moments, and time-series properties.
"""

import pandas as pd
import numpy as np
from loguru import logger


class StatisticalFeatures:
    """Computes statistical features from price/return data."""

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"]
        volume = df.get("Volume")

        df["log_return"] = np.log(close / close.shift(1))
        df["simple_return"] = close.pct_change()
        log_ret = df["log_return"]

        # Rolling distribution moments
        for w in [5, 10, 20, 60]:
            df[f"ret_mean_{w}"] = log_ret.rolling(w).mean()
            df[f"ret_std_{w}"] = log_ret.rolling(w).std()
            df[f"ret_skew_{w}"] = log_ret.rolling(w).skew()
            df[f"ret_kurt_{w}"] = log_ret.rolling(w).kurt()
            df[f"ann_vol_{w}"] = df[f"ret_std_{w}"] * np.sqrt(252)
            df[f"ret_to_vol_{w}"] = df[f"ret_mean_{w}"] / (df[f"ret_std_{w}"] + 1e-10)
            df[f"ret_max_{w}"] = log_ret.rolling(w).max()
            df[f"ret_min_{w}"] = log_ret.rolling(w).min()
            df[f"ret_range_{w}"] = df[f"ret_max_{w}"] - df[f"ret_min_{w}"]

        # Autocorrelation at various lags
        for lag in [1, 2, 5, 10, 20]:
            df[f"autocorr_{lag}"] = log_ret.rolling(60).apply(
                lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else 0, raw=True
            )

        # Drawdown features
        rolling_max = close.expanding().max()
        df["drawdown"] = (close - rolling_max) / rolling_max
        df["drawdown_duration"] = StatisticalFeatures._drawdown_duration(close)
        for w in [20, 60]:
            rm = close.rolling(w).max()
            df[f"drawdown_{w}"] = (close - rm) / rm

        # Volatility regime (short/long vol ratio)
        if "ret_std_5" in df.columns and "ret_std_60" in df.columns:
            df["vol_ratio_5_60"] = df["ret_std_5"] / (df["ret_std_60"] + 1e-10)

        # Hurst exponent approximation
        df["hurst_60"] = log_ret.rolling(60).apply(StatisticalFeatures._hurst, raw=True)

        # Volume statistics
        if volume is not None:
            log_vol = np.log1p(volume)
            for w in [5, 20]:
                df[f"vol_mean_{w}"] = log_vol.rolling(w).mean()
                df[f"vol_std_{w}"] = log_vol.rolling(w).std()
            for w in [20, 60]:
                df[f"price_vol_corr_{w}"] = close.rolling(w).corr(volume)

        # Cumulative returns
        for w in [5, 10, 20, 60, 120, 252]:
            df[f"cum_return_{w}"] = close.pct_change(periods=w)

        # Price z-scores
        for w in [20, 60, 120]:
            rm = close.rolling(w).mean()
            rs = close.rolling(w).std()
            df[f"price_zscore_{w}"] = (close - rm) / (rs + 1e-10)

        # Volume z-scores
        if volume is not None:
            for w in [20, 60]:
                vm = volume.rolling(w).mean()
                vs = volume.rolling(w).std()
                df[f"volume_zscore_{w}"] = (volume - vm) / (vs + 1e-10)

        logger.info(f"Computed statistical features, total columns: {len(df.columns)}")
        return df

    @staticmethod
    def _drawdown_duration(prices: pd.Series) -> pd.Series:
        rolling_max = prices.expanding().max()
        in_dd = prices < rolling_max
        duration = pd.Series(0, index=prices.index, dtype=int)
        count = 0
        for i in range(len(in_dd)):
            count = count + 1 if in_dd.iloc[i] else 0
            duration.iloc[i] = count
        return duration

    @staticmethod
    def _hurst(series: np.ndarray) -> float:
        if len(series) < 20:
            return 0.5
        try:
            n = len(series)
            max_k = min(n // 2, 50)
            if max_k < 4:
                return 0.5
            rs_vals, ns = [], []
            for k in range(4, max_k + 1):
                sub = series[:k]
                dev = np.cumsum(sub - np.mean(sub))
                r = np.max(dev) - np.min(dev)
                s = np.std(sub, ddof=1)
                if s > 0:
                    rs_vals.append(r / s)
                    ns.append(k)
            if len(rs_vals) < 3:
                return 0.5
            return float(np.clip(np.polyfit(np.log(ns), np.log(rs_vals), 1)[0], 0, 1))
        except Exception:
            return 0.5
