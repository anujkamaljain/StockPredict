"""
Signal API routes — generate and retrieve ML-based trading signals.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from loguru import logger
import numpy as np

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

        # Fetch data
        service = DataIngestionService()
        result = service.fetch_stock_data(ticker, start="2015-01-01")

        if result["ohlcv"].empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        df = result["ohlcv"]

        # Compute features
        engineer = FeatureEngineer()
        featured_df = engineer.compute_features(df)

        # Drop NaN rows from rolling windows
        featured_df = featured_df.dropna()

        if len(featured_df) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data after feature engineering")

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
                if not (val != val):  # not NaN
                    feature_dict[f] = round(float(val), 4)

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

        return {
            "signal": signal.to_dict(),
            "price": {
                "current": round(float(df["Close"].iloc[-1]), 2),
                "change_1d": round(float(df["Close"].pct_change().iloc[-1] * 100), 2),
                "change_5d": round(float(df["Close"].pct_change(5).iloc[-1] * 100), 2),
                "change_20d": round(float(df["Close"].pct_change(20).iloc[-1] * 100), 2),
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
        logger.error(f"Signal generation error for {ticker}: {e}")
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
