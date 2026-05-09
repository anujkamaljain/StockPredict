"""
Technical indicator computation.
Uses ta library and custom implementations for advanced indicators.
"""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


class TechnicalIndicators:
    """
    Computes a comprehensive set of technical indicators from OHLCV data.
    All indicators are computed without look-ahead bias.
    """

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators and append as new columns.

        Args:
            df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)

        Returns:
            DataFrame with original + indicator columns
        """
        df = df.copy()

        # Ensure required columns exist
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns for technical indicators: {missing}")
            return df

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        open_ = df["Open"]

        # =====================================================================
        # TREND INDICATORS
        # =====================================================================

        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{period}"] = close.rolling(window=period).mean()

        # Exponential Moving Averages
        for period in [9, 12, 21, 26, 50]:
            df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # SMA crossover signals
        df["sma_cross_20_50"] = (df["sma_20"] - df["sma_50"]) / df["sma_50"]
        df["sma_cross_50_200"] = (df["sma_50"] - df["sma_200"]) / df["sma_200"]

        # Price relative to SMAs
        for period in [20, 50, 200]:
            df[f"price_to_sma_{period}"] = close / df[f"sma_{period}"] - 1

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # ADX (Average Directional Index)
        df = TechnicalIndicators._compute_adx(df, period=14)

        # Parabolic SAR approximation
        df["psar_trend"] = np.where(close > df["sma_20"], 1, -1)

        # =====================================================================
        # MOMENTUM INDICATORS
        # =====================================================================

        # RSI (Relative Strength Index) — multiple periods
        for period in [7, 14, 21]:
            df[f"rsi_{period}"] = TechnicalIndicators._compute_rsi(close, period)

        # Stochastic Oscillator
        for period in [14]:
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            df[f"stoch_k_{period}"] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
            df[f"stoch_d_{period}"] = df[f"stoch_k_{period}"].rolling(window=3).mean()

        # Williams %R
        period = 14
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        df["williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = close.pct_change(periods=period) * 100

        # Momentum
        for period in [10, 20]:
            df[f"momentum_{period}"] = close - close.shift(period)

        # CCI (Commodity Channel Index)
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        df["cci_20"] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)

        # =====================================================================
        # VOLATILITY INDICATORS
        # =====================================================================

        # Bollinger Bands
        for period in [20]:
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            df[f"bb_upper_{period}"] = sma + 2 * std
            df[f"bb_lower_{period}"] = sma - 2 * std
            df[f"bb_width_{period}"] = (df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]) / sma
            df[f"bb_pct_{period}"] = (close - df[f"bb_lower_{period}"]) / (
                df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"] + 1e-10
            )

        # ATR (Average True Range)
        for period in [14, 21]:
            df[f"atr_{period}"] = TechnicalIndicators._compute_atr(df, period)
            df[f"atr_pct_{period}"] = df[f"atr_{period}"] / close

        # Keltner Channels
        ema_20 = close.ewm(span=20, adjust=False).mean()
        atr_10 = TechnicalIndicators._compute_atr(df, 10)
        df["keltner_upper"] = ema_20 + 2 * atr_10
        df["keltner_lower"] = ema_20 - 2 * atr_10
        df["keltner_pct"] = (close - df["keltner_lower"]) / (
            df["keltner_upper"] - df["keltner_lower"] + 1e-10
        )

        # Donchian Channel
        for period in [20]:
            df[f"donchian_high_{period}"] = high.rolling(window=period).max()
            df[f"donchian_low_{period}"] = low.rolling(window=period).min()
            df[f"donchian_mid_{period}"] = (
                df[f"donchian_high_{period}"] + df[f"donchian_low_{period}"]
            ) / 2

        # Historical Volatility
        log_returns = np.log(close / close.shift(1))
        for period in [10, 20, 60]:
            df[f"volatility_{period}"] = log_returns.rolling(window=period).std() * np.sqrt(252)

        # =====================================================================
        # VOLUME INDICATORS
        # =====================================================================

        # OBV (On-Balance Volume)
        obv = pd.Series(0, index=df.index, dtype=float)
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        df["obv"] = obv
        df["obv_sma_20"] = df["obv"].rolling(window=20).mean()

        # Volume SMA ratio
        for period in [20, 50]:
            vol_sma = volume.rolling(window=period).mean()
            df[f"volume_ratio_{period}"] = volume / (vol_sma + 1e-10)

        # VWAP (Volume Weighted Average Price) - rolling
        typical = (high + low + close) / 3
        cumulative_tp_vol = (typical * volume).rolling(window=20).sum()
        cumulative_vol = volume.rolling(window=20).sum()
        df["vwap_20"] = cumulative_tp_vol / (cumulative_vol + 1e-10)
        df["price_to_vwap"] = close / df["vwap_20"] - 1

        # MFI (Money Flow Index)
        df["mfi_14"] = TechnicalIndicators._compute_mfi(df, period=14)

        # A/D Line (Accumulation/Distribution)
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        df["ad_line"] = (clv * volume).cumsum()

        # CMF (Chaikin Money Flow)
        mf_volume = clv * volume
        df["cmf_20"] = mf_volume.rolling(window=20).sum() / (volume.rolling(window=20).sum() + 1e-10)

        # =====================================================================
        # CANDLESTICK FEATURES
        # =====================================================================

        # Candle body and shadow ratios
        body = abs(close - open_)
        full_range = high - low + 1e-10
        df["candle_body_pct"] = body / full_range
        df["upper_shadow_pct"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / full_range
        df["lower_shadow_pct"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / full_range
        df["candle_direction"] = np.where(close >= open_, 1, -1)

        # Gap
        df["gap_pct"] = (open_ - close.shift(1)) / (close.shift(1) + 1e-10)

        logger.info(f"Computed {len(df.columns) - 5} technical indicators")
        return df

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI using exponential moving average of gains/losses."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.ewm(span=period, adjust=False).mean()
        return atr

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Average Directional Index (ADX)."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = TechnicalIndicators._compute_atr(df, period)

        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10))

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()

        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        df["adx"] = adx

        return df

    @staticmethod
    def _compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Money Flow Index."""
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        money_flow = typical_price * df["Volume"]

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0.0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0.0)

        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_sum / (negative_sum + 1e-10)))
        return mfi
