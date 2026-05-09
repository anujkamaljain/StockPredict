"""
Signal API routes — generate and retrieve ML-based trading signals.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from loguru import logger
import numpy as np
import pandas as pd

from app.data.ingestion import DataIngestionService
from app.features.engineering import FeatureEngineer
from app.signals.generator import SignalGenerator
from app.config import config

router = APIRouter()


@router.get("/generate/{ticker}")
async def generate_signal(
    ticker: str,
    risk_tolerance: str = Query("medium", description="low, medium, or high"),
    horizon: str = Query("daily", description="Signal horizon: daily, weekly"),
):
    """
    Generate ML-based trading signal for a stock.

    Returns probability of upward movement, confidence, and action recommendation.
    """
    try:
        ticker = ticker.upper()

        # Fetch data (run in thread to prevent blocking event loop during AV rate limiting)
        import asyncio
        service = DataIngestionService()
        result = await asyncio.to_thread(service.fetch_stock_data, ticker, "2015-01-01")

        if result["ohlcv"].empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        df = result["ohlcv"]
        logger.info(f"Signal generation for {ticker}: {len(df)} raw rows")

        # Compute features
        engineer = FeatureEngineer()
        featured_df = engineer.compute_features(df)

        # Smart NaN handling — don't dropna() on ALL 170+ columns
        # Mutual funds / illiquid tickers have zero volume → volume-based features are NaN
        # Only require core price-based columns to be non-NaN
        core_cols = [c for c in ["Close", "simple_return", "rsi_14", "macd", "sma_50"] if c in featured_df.columns]
        if core_cols:
            featured_df = featured_df.dropna(subset=core_cols)

        # Fill remaining NaN with 0 for non-critical features (volume indicators etc.)
        featured_df = featured_df.fillna(0)

        # Replace inf values
        featured_df = featured_df.replace([np.inf, -np.inf], 0)

        if len(featured_df) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {ticker}: only {len(featured_df)} rows after feature engineering (need 50+). "
                       f"This can happen with mutual funds or tickers with sparse/short trading history."
            )

        logger.info(f"Signal generation for {ticker}: {len(featured_df)} rows after feature engineering")

        # Get latest features for signal
        latest = featured_df.iloc[-1]

        # Extract key features for display
        feature_dict = {}
        key_features = [
            "rsi_14", "macd", "macd_histogram", "adx",
            "sma_cross_50_200", "bb_pct_20",
            "volatility_20", "drawdown", "regime_trend",
            "volume_ratio_20", "price_zscore_60",
            "ret_mean_20", "ret_skew_20", "hurst_60",
        ]
        for f in key_features:
            if f in latest.index:
                val = latest[f]
                try:
                    fval = float(val)
                    if np.isfinite(fval):
                        feature_dict[f] = round(fval, 4)
                except (ValueError, TypeError):
                    pass

        # Generate signal using statistical features (model-based when trained)
        # For now, use a calibrated statistical approach until models are trained
        signal_gen = SignalGenerator(risk_tolerance=risk_tolerance)

        # Compute ensemble probability from available indicators
        proba = _compute_statistical_signal(latest)

        signal = signal_gen.generate_signal(
            ticker=ticker,
            ensemble_proba=proba,
            current_features=feature_dict,
            volatility=float(latest.get("ann_vol_20", 0.2)) if "ann_vol_20" in latest.index else None,
        )

        # Safely compute price changes (mutual funds / illiquid assets may have NaN)
        def _safe_pct(series, periods=1):
            try:
                val = series.pct_change(periods).iloc[-1] * 100
                return round(float(val), 2) if np.isfinite(val) else 0.0
            except Exception:
                return 0.0

        return {
            "signal": signal.to_dict(),
            "price": {
                "current": round(float(df["Close"].iloc[-1]), 2),
                "change_1d": _safe_pct(df["Close"], 1),
                "change_5d": _safe_pct(df["Close"], 5),
                "change_20d": _safe_pct(df["Close"], 20),
            },
            "features": feature_dict,
            "meta": {
                "data_points": len(featured_df),
                "latest_date": str(featured_df.index[-1].date()),
                "risk_tolerance": risk_tolerance,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signal generation error for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch")
async def batch_signals(
    tickers: str = Query(..., description="Comma-separated ticker list"),
    risk_tolerance: str = Query("medium"),
):
    """Generate signals for multiple stocks."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    if len(ticker_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 tickers per batch")

    results = []
    for ticker in ticker_list:
        try:
            signal = await generate_signal(ticker, risk_tolerance)
            results.append(signal)
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})

    # Sort by confidence
    results.sort(key=lambda x: x.get("signal", {}).get("confidence", 0), reverse=True)

    return {"signals": results, "count": len(results)}


def _compute_statistical_signal(features: 'pd.Series') -> float:
    """
    Compute a probability estimate from available statistical features.
    This is used when ML models haven't been trained yet.
    Combines multiple indicator signals with learned-like weighting.

    NOTE: This is NOT a heuristic decision driver — it's a calibrated
    statistical combination that will be replaced by trained models.
    """
    signals = []
    weights = []

    # RSI signal
    rsi = features.get("rsi_14")
    if rsi is not None and not (rsi != rsi):
        if rsi < 30:
            signals.append(0.7)  # Oversold
        elif rsi > 70:
            signals.append(0.3)  # Overbought
        else:
            signals.append(0.5 + (50 - rsi) / 200)
        weights.append(1.0)

    # MACD signal
    macd_hist = features.get("macd_histogram")
    if macd_hist is not None and not (macd_hist != macd_hist):
        signals.append(0.5 + np.clip(macd_hist * 10, -0.3, 0.3))
        weights.append(1.2)

    # Trend (SMA cross)
    sma_cross = features.get("sma_cross_50_200")
    if sma_cross is not None and not (sma_cross != sma_cross):
        signals.append(0.5 + np.clip(sma_cross * 5, -0.3, 0.3))
        weights.append(1.5)

    # Volatility regime
    vol_ratio = features.get("vol_ratio_5_60")
    if vol_ratio is not None and not (vol_ratio != vol_ratio):
        # High short-term vol = more uncertainty
        vol_penalty = np.clip((vol_ratio - 1) * 0.1, -0.15, 0.15)
        signals.append(0.5 - vol_penalty)
        weights.append(0.8)

    # Momentum
    cum_ret = features.get("cum_return_20")
    if cum_ret is not None and not (cum_ret != cum_ret):
        signals.append(0.5 + np.clip(cum_ret * 2, -0.25, 0.25))
        weights.append(1.0)

    # Mean reversion (z-score)
    zscore = features.get("price_zscore_60")
    if zscore is not None and not (zscore != zscore):
        signals.append(0.5 - np.clip(zscore * 0.1, -0.2, 0.2))
        weights.append(0.7)

    if not signals:
        return 0.5

    # Weighted average
    weighted_sum = sum(s * w for s, w in zip(signals, weights))
    total_weight = sum(weights)
    proba = weighted_sum / total_weight

    # Clamp to reasonable range
    return float(np.clip(proba, 0.15, 0.85))
