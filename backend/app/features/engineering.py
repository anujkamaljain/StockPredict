"""
Feature engineering pipeline orchestrator.
Combines technical, statistical, regime, and time-based features.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from loguru import logger
from sklearn.preprocessing import RobustScaler
import joblib
from pathlib import Path

from app.features.technical import TechnicalIndicators
from app.features.statistical import StatisticalFeatures
from app.features.regime import RegimeDetector


class FeatureEngineer:
    """
    Master feature engineering pipeline.
    Orchestrates all feature computation and preprocessing.
    """

    def __init__(self, scaler_path: Optional[Path] = None):
        self.technical = TechnicalIndicators()
        self.statistical = StatisticalFeatures()
        self.regime = RegimeDetector()
        self.scaler = RobustScaler()  # Robust to outliers
        self.scaler_path = scaler_path
        self.feature_names: List[str] = []
        self._fitted = False

    def compute_features(
        self,
        df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute all features from raw OHLCV data.

        Args:
            df: OHLCV DataFrame for a single stock
            macro_df: Macroeconomic data from FRED
            benchmark_df: Benchmark index data (e.g., SPY)

        Returns:
            DataFrame with all features computed
        """
        logger.info(f"Computing features for {len(df)} rows...")

        # Step 1: Technical indicators
        df = self.technical.compute_all(df)

        # Step 2: Statistical features
        df = self.statistical.compute_all(df)

        # Step 3: Regime features
        df = self.regime.compute_regime_features(df)

        # Step 4: Time-based features
        df = self._add_time_features(df)

        # Step 5: Lag features (critical for avoiding leakage)
        df = self._add_lag_features(df)

        # Step 6: Merge macro data
        if macro_df is not None and not macro_df.empty:
            df = self._merge_macro(df, macro_df)

        # Step 7: Relative strength vs benchmark
        if benchmark_df is not None and not benchmark_df.empty:
            df = self._add_relative_strength(df, benchmark_df)

        # Step 8: Target variable (next-day return direction)
        df = self._create_targets(df)

        logger.info(f"Total features computed: {len(df.columns)}")
        return df

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_col: str = "target_direction",
        drop_cols: Optional[List[str]] = None,
        fit_scaler: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for model training.

        Returns:
            (X, y, feature_names) tuple
        """
        # Drop NaN rows created by rolling windows
        df = df.dropna()

        # Define columns to exclude from features
        exclude_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "target_return", "target_direction", "target_return_5d",
            "target_direction_5d", "target_volatility_5d",
        ]
        if drop_cols:
            exclude_cols.extend(drop_cols)

        # Feature columns
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df[target_col].values if target_col in df.columns else None

        # Replace inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        if fit_scaler:
            X = self.scaler.fit_transform(X)
            self._fitted = True
            if self.scaler_path:
                joblib.dump(self.scaler, self.scaler_path)
                logger.info(f"Scaler saved to {self.scaler_path}")
        elif self._fitted:
            X = self.scaler.transform(X)

        return X, y, feature_cols

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call prepare_for_training first.")

        feature_cols = [c for c in self.feature_names if c in df.columns]
        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X)

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar and seasonality features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        df["day_of_week"] = df.index.dayofweek / 4.0  # Normalize to [0, 1]
        df["day_of_month"] = df.index.day / 31.0
        df["month"] = df.index.month / 12.0
        df["quarter"] = df.index.quarter / 4.0
        df["week_of_year"] = df.index.isocalendar().week.values / 52.0

        # Cyclical encoding for periodicity
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
        df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)

        # Month-end / quarter-end effects
        df["is_month_end"] = df.index.is_month_end.astype(float)
        df["is_month_start"] = df.index.is_month_start.astype(float)
        df["is_quarter_end"] = df.index.is_quarter_end.astype(float)

        # Days since year start (January effect)
        df["day_of_year"] = df.index.dayofyear / 365.0

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features to capture temporal dependencies."""
        if "simple_return" not in df.columns:
            return df

        # Lagged returns
        for lag in [1, 2, 3, 5, 10]:
            df[f"return_lag_{lag}"] = df["simple_return"].shift(lag)

        # Lagged volume change
        if "Volume" in df.columns:
            vol_change = df["Volume"].pct_change()
            for lag in [1, 2, 5]:
                df[f"vol_change_lag_{lag}"] = vol_change.shift(lag)

        # Lagged volatility
        if "ret_std_5" in df.columns:
            for lag in [1, 5]:
                df[f"vol5_lag_{lag}"] = df["ret_std_5"].shift(lag)

        return df

    def _merge_macro(self, df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Merge macroeconomic features with stock data."""
        # Reindex macro data to match stock dates
        macro_aligned = macro_df.reindex(df.index, method="ffill")

        # Add prefix to avoid column name conflicts
        macro_aligned.columns = [f"macro_{c}" for c in macro_aligned.columns]

        # Only add columns that don't already exist
        new_cols = [c for c in macro_aligned.columns if c not in df.columns]
        if new_cols:
            df = pd.concat([df, macro_aligned[new_cols]], axis=1)

        logger.info(f"Merged {len(new_cols)} macro features")
        return df

    def _add_relative_strength(
        self, df: pd.DataFrame, benchmark_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute relative strength vs benchmark."""
        if "Close" not in benchmark_df.columns:
            return df

        bench_close = benchmark_df["Close"].reindex(df.index, method="ffill")
        stock_ret = df["Close"].pct_change()
        bench_ret = bench_close.pct_change()

        # Relative strength
        df["relative_strength"] = stock_ret - bench_ret

        # Rolling relative performance
        for w in [5, 20, 60]:
            stock_cum = (1 + stock_ret).rolling(w).apply(np.prod, raw=True) - 1
            bench_cum = (1 + bench_ret).rolling(w).apply(np.prod, raw=True) - 1
            df[f"relative_perf_{w}"] = stock_cum - bench_cum

        # Beta (rolling)
        for w in [60, 252]:
            cov = stock_ret.rolling(w).cov(bench_ret)
            var = bench_ret.rolling(w).var()
            df[f"beta_{w}"] = cov / (var + 1e-10)

        # Alpha (rolling Jensen's alpha)
        rf_daily = 0.05 / 252  # Approximate
        for w in [60, 252]:
            if f"beta_{w}" in df.columns:
                expected = rf_daily + df[f"beta_{w}"] * (bench_ret - rf_daily)
                df[f"alpha_{w}"] = stock_ret.rolling(w).mean() - expected.rolling(w).mean()

        return df

    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create prediction target variables (shifted forward to avoid leakage)."""
        close = df["Close"]

        # Next-day return (primary target)
        df["target_return"] = close.shift(-1) / close - 1
        df["target_direction"] = (df["target_return"] > 0).astype(int)

        # 5-day forward return
        df["target_return_5d"] = close.shift(-5) / close - 1
        df["target_direction_5d"] = (df["target_return_5d"] > 0).astype(int)

        # Forward volatility (for risk prediction)
        df["target_volatility_5d"] = df["log_return"].shift(-1).rolling(5).std() * np.sqrt(252)

        return df
