"""
Market regime detection using Hidden Markov Models.
Classifies market states: bull, bear, high-volatility, low-volatility.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class RegimeDetector:
    """
    Detects market regimes using statistical methods and HMM.
    Provides regime labels as features for ML models.
    """

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None

    def compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute regime-related features without requiring HMM fitting."""
        df = df.copy()
        close = df["Close"]
        returns = close.pct_change()

        # Trend regime: SMA-based
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        df["regime_trend"] = np.where(
            sma_50 > sma_200, 1, np.where(sma_50 < sma_200, -1, 0)
        ).astype(float)

        # Volatility regime: rolling vol percentile
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_percentile = vol_20.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=True
        )
        df["regime_vol_percentile"] = vol_percentile
        df["regime_high_vol"] = (vol_percentile > 0.8).astype(float)
        df["regime_low_vol"] = (vol_percentile < 0.2).astype(float)

        # Momentum regime
        mom_20 = returns.rolling(20).sum()
        mom_60 = returns.rolling(60).sum()
        df["regime_momentum_20"] = np.sign(mom_20).astype(float)
        df["regime_momentum_60"] = np.sign(mom_60).astype(float)

        # Mean-reversion vs trend score (based on Hurst-like metric)
        df["regime_mean_revert_score"] = returns.rolling(60).apply(
            self._mean_reversion_score, raw=True
        )

        # Breadth approximation using price vs multiple SMAs
        above_count = 0
        for p in [10, 20, 50, 100, 200]:
            sma = close.rolling(p).mean()
            above_count = above_count + (close > sma).astype(int)
        df["regime_sma_breadth"] = above_count / 5.0

        logger.info("Computed regime features")
        return df

    def fit_hmm(self, returns: pd.Series) -> Optional[np.ndarray]:
        """Fit Hidden Markov Model to classify regimes."""
        try:
            from hmmlearn.hmm import GaussianHMM

            clean_returns = returns.dropna().values.reshape(-1, 1)
            if len(clean_returns) < 100:
                logger.warning("Insufficient data for HMM fitting")
                return None

            model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            model.fit(clean_returns)
            self.model = model

            states = model.predict(clean_returns)

            # Order states by mean return (0=bear, 1=neutral, 2=bull)
            state_means = [clean_returns[states == i].mean() for i in range(self.n_regimes)]
            order = np.argsort(state_means)
            state_map = {old: new for new, old in enumerate(order)}
            ordered_states = np.array([state_map[s] for s in states])

            return ordered_states

        except ImportError:
            logger.warning("hmmlearn not installed, skipping HMM regime detection")
            return None
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return None

    @staticmethod
    def _mean_reversion_score(returns: np.ndarray) -> float:
        """Score how mean-reverting the series is (-1 to 1)."""
        if len(returns) < 10:
            return 0.0
        try:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            return -autocorr  # Negative autocorrelation = mean reversion
        except Exception:
            return 0.0
