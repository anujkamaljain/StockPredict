"""
Data ingestion orchestrator.
Coordinates data fetching from multiple providers with fallback logic.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
from datetime import datetime

from app.data.providers.yahoo import YahooFinanceProvider
from app.data.providers.alpha_vantage import AlphaVantageProvider
from app.data.providers.fred import FREDProvider
from app.data.validation import DataValidator, ValidationReport
from app.config import config


class DataIngestionService:
    """
    Orchestrates data fetching from multiple providers.
    Implements fallback logic and data quality validation.
    """

    def __init__(self):
        self.yahoo = YahooFinanceProvider(cache_dir=config.data.cache_dir / "yahoo")
        self.alpha_vantage = AlphaVantageProvider(
            api_key=config.api_keys.alpha_vantage,
            cache_dir=config.data.cache_dir / "alpha_vantage"
        )
        self.fred = FREDProvider(
            api_key=config.api_keys.fred,
            cache_dir=config.data.cache_dir / "fred"
        )
        self.validator = DataValidator()

    def fetch_stock_data(
        self,
        ticker: str,
        start: str = "2010-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
        validate: bool = True,
        clean: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch complete stock data (OHLCV + fundamentals) with fallback.

        Priority:
        🥇 1. Alpha Vantage (primary — official API, stable, OHLCV + indicators)
        🥈 2. Yahoo Finance (fallback — missing data recovery, sanity checks)

        Returns:
            Dict with 'ohlcv', 'fundamentals', 'validation_report'
        """
        result = {
            "ticker": ticker,
            "ohlcv": pd.DataFrame(),
            "fundamentals": {},
            "validation_report": None,
            "source": None,
        }

        ohlcv = pd.DataFrame()

        # 🥇 Try Alpha Vantage first (primary source)
        if self.alpha_vantage.is_configured and self.alpha_vantage.requests_remaining > 0:
            logger.info(f"Fetching {ticker} data from Alpha Vantage (primary)...")
            try:
                ohlcv = self.alpha_vantage.fetch_daily(ticker)
                if not ohlcv.empty:
                    result["source"] = "alpha_vantage"
                    logger.info(f"Alpha Vantage returned {len(ohlcv)} rows for {ticker}")
                else:
                    logger.warning(f"Alpha Vantage returned empty data for {ticker}")
            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {ticker}: {e}")
        else:
            if not self.alpha_vantage.is_configured:
                logger.info(f"Alpha Vantage not configured, skipping to Yahoo fallback for {ticker}")
            else:
                logger.warning(f"Alpha Vantage rate limit reached, falling back to Yahoo for {ticker}")

        # 🥈 Fallback to Yahoo Finance
        if ohlcv.empty:
            logger.info(f"Falling back to Yahoo Finance for {ticker}...")
            ohlcv = self.yahoo.fetch_ohlcv(ticker, start=start, end=end, interval=interval)
            if not ohlcv.empty:
                result["source"] = "yahoo"
                logger.info(f"Yahoo Finance returned {len(ohlcv)} rows for {ticker}")
            else:
                logger.error(f"All data sources exhausted for {ticker}")
                return result

        # If Alpha Vantage was primary and start date filtering is needed
        if result["source"] == "alpha_vantage" and start:
            ohlcv = ohlcv[ohlcv.index >= pd.Timestamp(start)]

        # Validate data
        if validate and not ohlcv.empty:
            report = self.validator.validate(ohlcv, ticker)
            result["validation_report"] = report

            if not report.is_valid:
                logger.warning(f"Validation issues for {ticker}: {report.issues}")

        # Clean data
        if clean and not ohlcv.empty:
            ohlcv = self.validator.clean(ohlcv, ticker)

        result["ohlcv"] = ohlcv

        # Fetch fundamentals — skip AV if it already failed for this ticker's OHLCV
        try:
            if result["source"] == "alpha_vantage" and self.alpha_vantage.requests_remaining > 0:
                # AV worked for OHLCV, so try AV for fundamentals too
                overview = self.alpha_vantage.fetch_company_overview(ticker)
                if overview:
                    result["fundamentals"] = overview
                else:
                    result["fundamentals"] = self.yahoo.fetch_fundamentals(ticker)
            else:
                # AV failed or wasn't used — go straight to Yahoo for fundamentals
                result["fundamentals"] = self.yahoo.fetch_fundamentals(ticker)
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")

        return result

    def fetch_universe(
        self,
        tickers: Optional[List[str]] = None,
        start: str = "2010-01-01",
        end: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch data for an entire stock universe.

        Args:
            tickers: List of ticker symbols. Defaults to config.default_tickers.
            start: Start date
            end: End date

        Returns:
            Dict mapping ticker -> {ohlcv, fundamentals, validation_report}
        """
        if tickers is None:
            tickers = config.default_tickers

        logger.info(f"Fetching data for {len(tickers)} tickers...")
        universe = {}

        for i, ticker in enumerate(tickers):
            logger.info(f"[{i + 1}/{len(tickers)}] Processing {ticker}")
            data = self.fetch_stock_data(ticker, start=start, end=end)

            if not data["ohlcv"].empty:
                universe[ticker] = data

                # Save to Parquet
                parquet_path = config.data.raw_dir / f"{ticker}.parquet"
                data["ohlcv"].to_parquet(parquet_path)
            else:
                logger.warning(f"Skipping {ticker} — no valid data")

        logger.info(f"Successfully fetched data for {len(universe)}/{len(tickers)} tickers")
        return universe

    def fetch_benchmark(
        self,
        start: str = "2010-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch benchmark index data (default: SPY)."""
        result = self.fetch_stock_data(
            config.benchmark_ticker, start=start, end=end
        )
        return result["ohlcv"]

    def fetch_macro_data(
        self,
        start: str = "2005-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch macroeconomic indicators from FRED."""
        return self.fred.fetch_all_macro_indicators(start=start, end=end)

    def load_cached_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load previously fetched data from Parquet cache."""
        parquet_path = config.data.raw_dir / f"{ticker}.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        return None

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of all cached data."""
        raw_files = list(config.data.raw_dir.glob("*.parquet"))
        summary = {
            "total_tickers": len(raw_files),
            "tickers": [],
        }

        for f in raw_files:
            ticker = f.stem
            df = pd.read_parquet(f)
            summary["tickers"].append({
                "ticker": ticker,
                "rows": len(df),
                "start": df.index.min().strftime("%Y-%m-%d") if len(df) > 0 else None,
                "end": df.index.max().strftime("%Y-%m-%d") if len(df) > 0 else None,
                "file_size_mb": f.stat().st_size / (1024 * 1024),
            })

        return summary
