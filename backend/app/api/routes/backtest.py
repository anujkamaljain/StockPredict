"""
Backtest API routes — run backtests and retrieve results.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from loguru import logger
import numpy as np

from app.data.ingestion import DataIngestionService
from app.features.engineering import FeatureEngineer
from app.backtest.engine import BacktestEngine
from app.signals.generator import SignalGenerator

router = APIRouter()


@router.get("/run/{ticker}")
async def run_backtest(
    ticker: str,
    start: str = Query("2018-01-01"),
    end: Optional[str] = Query(None),
    initial_capital: float = Query(100000),
    risk_tolerance: str = Query("medium"),
    transaction_cost: float = Query(0.001),
    stop_loss: float = Query(0.05),
):
    """
    Run backtest for a single stock.

    Uses the signal generator to produce buy/sell/hold signals,
    then simulates trading with realistic constraints.
    """
    try:
        ticker = ticker.upper()

        # Fetch data (run in thread to prevent blocking event loop)
        import asyncio
        service = DataIngestionService()
        result = await asyncio.to_thread(service.fetch_stock_data, ticker, start, end)

        if result["ohlcv"].empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        df = result["ohlcv"]

        # Compute features
        engineer = FeatureEngineer()
        featured_df = engineer.compute_features(df)

        # Smart NaN handling — only require core columns
        core_cols = [c for c in ["Close", "simple_return", "rsi_14", "macd", "sma_50"] if c in featured_df.columns]
        if core_cols:
            featured_df = featured_df.dropna(subset=core_cols)
        featured_df = featured_df.fillna(0)
        featured_df = featured_df.replace([np.inf, -np.inf], 0)

        if len(featured_df) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {ticker}: {len(featured_df)} rows after feature engineering (need 50+)"
            )

        # Generate signals
        signal_gen = SignalGenerator(risk_tolerance=risk_tolerance)
        signals = np.zeros(len(featured_df))
        confidence = np.zeros(len(featured_df))

        from app.api.routes.signals import _compute_statistical_signal

        for i in range(len(featured_df)):
            row = featured_df.iloc[i]
            proba = _compute_statistical_signal(row)
            conf = abs(proba - 0.5) * 2

            if proba > signal_gen.buy_threshold and conf >= signal_gen.min_confidence:
                signals[i] = 1
            elif proba < signal_gen.sell_threshold and conf >= signal_gen.min_confidence:
                signals[i] = -1

            confidence[i] = conf

        # Run backtest
        engine = BacktestEngine(
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost,
            stop_loss_pct=stop_loss,
        )

        bt_result = engine.run(
            prices=featured_df["Close"],
            signals=signals.astype(int),
            confidence=confidence,
            dates=featured_df.index,
            ticker=ticker,
        )

        return {
            "ticker": ticker,
            "period": {
                "start": str(featured_df.index[0].date()),
                "end": str(featured_df.index[-1].date()),
            },
            "metrics": bt_result["metrics"],
            "buy_hold_metrics": bt_result["buy_hold_metrics"],
            "equity_curve": bt_result["equity_curve"],
            "trades": bt_result["trades"][:50],  # Limit for response size
            "total_trades": len(bt_result["trades"]),
            "initial_capital": bt_result["initial_capital"],
            "final_capital": round(bt_result["final_capital"], 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
