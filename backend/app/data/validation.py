"""
Data validation and cleaning pipeline.
Ensures data quality before feature engineering and model training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from loguru import logger
from dataclasses import dataclass


@dataclass
class ValidationReport:
    """Report from data validation."""
    ticker: str
    total_rows: int
    missing_pct: Dict[str, float]
    outlier_count: Dict[str, int]
    gaps: List[Tuple[str, str]]  # List of (start, end) gap periods
    is_valid: bool
    issues: List[str]


class DataValidator:
    """
    Validates and cleans market data to ensure quality for ML pipeline.
    Handles missing values, outliers, corporate actions, and data gaps.
    """

    def __init__(
        self,
        max_missing_pct: float = 0.05,
        max_gap_days: int = 5,
        outlier_std_threshold: float = 5.0,
        min_history_days: int = 252,  # 1 year of trading days
    ):
        self.max_missing_pct = max_missing_pct
        self.max_gap_days = max_gap_days
        self.outlier_std_threshold = outlier_std_threshold
        self.min_history_days = min_history_days

    def validate(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> ValidationReport:
        """
        Run full validation on OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            ticker: Ticker symbol for reporting

        Returns:
            ValidationReport with details
        """
        issues = []

        if df.empty:
            return ValidationReport(
                ticker=ticker, total_rows=0, missing_pct={}, outlier_count={},
                gaps=[], is_valid=False, issues=["Empty DataFrame"]
            )

        # Check minimum history
        if len(df) < self.min_history_days:
            issues.append(f"Insufficient history: {len(df)} rows < {self.min_history_days} required")

        # Check missing values
        missing_pct = (df.isnull().sum() / len(df)).to_dict()
        for col, pct in missing_pct.items():
            if pct > self.max_missing_pct:
                issues.append(f"High missing rate in {col}: {pct:.2%}")

        # Check for data gaps (weekends/holidays excluded)
        gaps = self._find_gaps(df)
        for start, end in gaps:
            issues.append(f"Data gap from {start} to {end}")

        # Check for outliers in returns
        outlier_count = {}
        if "Close" in df.columns:
            returns = df["Close"].pct_change().dropna()
            threshold = returns.std() * self.outlier_std_threshold
            outliers = (returns.abs() > threshold)
            outlier_count["Close_returns"] = int(outliers.sum())
            if outliers.sum() > 0:
                issues.append(f"Found {outliers.sum()} return outliers (>{self.outlier_std_threshold}σ)")

        # Check OHLCV consistency
        ohlcv_issues = self._check_ohlcv_consistency(df)
        issues.extend(ohlcv_issues)

        is_valid = len(issues) == 0 or all("outlier" in i.lower() or "gap" in i.lower() for i in issues)

        return ValidationReport(
            ticker=ticker,
            total_rows=len(df),
            missing_pct=missing_pct,
            outlier_count=outlier_count,
            gaps=gaps,
            is_valid=is_valid,
            issues=issues,
        )

    def _find_gaps(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find gaps in the time series larger than max_gap_days."""
        gaps = []
        if not isinstance(df.index, pd.DatetimeIndex):
            return gaps

        date_diffs = df.index.to_series().diff()
        # Filter for gaps > max_gap_days (excluding normal weekends)
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=self.max_gap_days)]

        for idx in large_gaps.index:
            prev_idx = df.index[df.index.get_loc(idx) - 1]
            gaps.append((prev_idx.strftime("%Y-%m-%d"), idx.strftime("%Y-%m-%d")))

        return gaps

    def _check_ohlcv_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check OHLCV data consistency."""
        issues = []
        required_cols = ["Open", "High", "Low", "Close"]

        present_cols = [c for c in required_cols if c in df.columns]
        if len(present_cols) < len(required_cols):
            missing = set(required_cols) - set(present_cols)
            issues.append(f"Missing columns: {missing}")
            return issues

        # High should be >= Open, Close, Low
        high_violations = (df["High"] < df[["Open", "Close", "Low"]].max(axis=1)).sum()
        if high_violations > 0:
            issues.append(f"High < max(O,C,L) in {high_violations} rows")

        # Low should be <= Open, Close, High
        low_violations = (df["Low"] > df[["Open", "Close", "High"]].min(axis=1)).sum()
        if low_violations > 0:
            issues.append(f"Low > min(O,C,H) in {low_violations} rows")

        # Volume should be non-negative
        if "Volume" in df.columns:
            neg_vol = (df["Volume"] < 0).sum()
            if neg_vol > 0:
                issues.append(f"Negative volume in {neg_vol} rows")

            zero_vol = (df["Volume"] == 0).sum()
            if zero_vol > len(df) * 0.1:
                issues.append(f"Zero volume in {zero_vol} rows ({zero_vol / len(df):.1%})")

        # Prices should be positive
        for col in present_cols:
            neg_prices = (df[col] <= 0).sum()
            if neg_prices > 0:
                issues.append(f"Non-positive {col} in {neg_prices} rows")

        return issues

    def clean(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> pd.DataFrame:
        """
        Clean OHLCV data.

        Operations:
        1. Remove duplicate indices
        2. Sort by date
        3. Forward-fill small gaps (up to max_gap_days)
        4. Winsorize extreme outliers in returns
        5. Fix OHLCV consistency violations
        6. Remove rows with zero volume (likely non-trading days)

        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df

        df = df.copy()

        # Remove duplicates
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            logger.info(f"[{ticker}] Removing {dup_count} duplicate indices")
            df = df[~df.index.duplicated(keep='last')]

        # Sort by date
        df.sort_index(inplace=True)

        # Forward-fill missing values (small gaps only)
        if df.isnull().any().any():
            missing_before = df.isnull().sum().sum()
            df = df.ffill(limit=self.max_gap_days)
            missing_after = df.isnull().sum().sum()
            logger.info(f"[{ticker}] Forward-filled {missing_before - missing_after} missing values")

        # Drop remaining NaN rows
        nan_rows = df.isnull().any(axis=1).sum()
        if nan_rows > 0:
            logger.info(f"[{ticker}] Dropping {nan_rows} rows with remaining NaN values")
            df = df.dropna()

        # Winsorize extreme returns (beyond 5 std devs)
        if "Close" in df.columns and len(df) > 1:
            returns = df["Close"].pct_change()
            mean_ret = returns.mean()
            std_ret = returns.std()
            lower_bound = mean_ret - self.outlier_std_threshold * std_ret
            upper_bound = mean_ret + self.outlier_std_threshold * std_ret

            extreme_returns = (returns < lower_bound) | (returns > upper_bound)
            if extreme_returns.sum() > 0:
                logger.info(f"[{ticker}] Winsorizing {extreme_returns.sum()} extreme returns")
                # Replace extreme values with previous close * (1 + clipped_return)
                clipped_returns = returns.clip(lower_bound, upper_bound)
                for idx in df.index[extreme_returns]:
                    loc = df.index.get_loc(idx)
                    if loc > 0:
                        prev_close = df["Close"].iloc[loc - 1]
                        df.loc[idx, "Close"] = prev_close * (1 + clipped_returns.loc[idx])

        # Fix OHLCV consistency
        if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
            df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)

        logger.info(f"[{ticker}] Cleaned data: {len(df)} rows remaining")
        return df
