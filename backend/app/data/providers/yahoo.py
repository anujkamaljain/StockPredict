"""
Yahoo Finance data provider using yfinance.
SECONDARY (fallback) data source — used when Alpha Vantage is unavailable,
for missing data recovery, sanity checks, and redundancy.
Also provides search autocomplete (unlimited, no API key needed).
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from loguru import logger
import json
from pathlib import Path


class YahooFinanceProvider:
    """
    Fallback data source using Yahoo Finance.
    Used for missing data recovery, sanity checks, and search autocomplete.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, ticker: str, data_type: str) -> Optional[Path]:
        """Get cache file path for a given ticker and data type."""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{ticker}_{data_type}.parquet"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 12) -> bool:
        """Check if cached data is still valid."""
        if not cache_path or not cache_path.exists():
            return False
        mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return (datetime.now() - mod_time) < timedelta(hours=max_age_hours)

    def fetch_ohlcv(
        self,
        ticker: str,
        start: str = "2010-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data.

        Args:
            ticker: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD), defaults to today
            interval: Data interval (1d, 1wk, 1mo)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data, DatetimeIndex
        """
        cache_path = self._get_cache_path(ticker, f"ohlcv_{interval}")

        if use_cache and cache_path and self._is_cache_valid(cache_path):
            logger.info(f"Loading cached OHLCV data for {ticker}")
            df = pd.read_parquet(cache_path)
            # Fetch only new data
            last_date = df.index.max()
            if end is None:
                end = datetime.now().strftime("%Y-%m-%d")
            if last_date.strftime("%Y-%m-%d") >= end:
                return df[start:end]
            # Fetch incremental data
            new_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            try:
                new_data = yf.download(ticker, start=new_start, end=end, interval=interval, progress=False)
                if not new_data.empty:
                    # Handle multi-level columns from yfinance
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data = new_data.droplevel(level=1, axis=1)
                    df = pd.concat([df, new_data])
                    df = df[~df.index.duplicated(keep='last')]
                    df.sort_index(inplace=True)
                    if cache_path:
                        df.to_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to fetch incremental data for {ticker}: {e}")
            return df[start:end]

        # Full fetch
        logger.info(f"Fetching OHLCV data for {ticker} from {start} to {end or 'now'}")
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(level=1, axis=1)

            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Cache the data
            if cache_path:
                df.to_parquet(cache_path)

            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a ticker.

        Returns dict with P/E, EPS, market cap, revenue, etc.
        """
        logger.info(f"Fetching fundamentals for {ticker}")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            fundamentals = {
                "ticker": ticker,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "forward_eps": info.get("forwardEps"),
                "pb_ratio": info.get("priceToBook"),
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "revenue": info.get("totalRevenue"),
                "revenue_growth": info.get("revenueGrowth"),
                "gross_margins": info.get("grossMargins"),
                "operating_margins": info.get("operatingMargins"),
                "profit_margins": info.get("profitMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "free_cash_flow": info.get("freeCashflow"),
                "beta": info.get("beta"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "50d_avg": info.get("fiftyDayAverage"),
                "200d_avg": info.get("twoHundredDayAverage"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "fetch_date": datetime.now().isoformat(),
            }

            # Cache fundamentals
            if self.cache_dir:
                cache_file = self.cache_dir / f"{ticker}_fundamentals.json"
                with open(cache_file, 'w') as f:
                    json.dump(fundamentals, f, indent=2, default=str)

            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    def fetch_dividends_splits(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Fetch dividend and stock split history."""
        try:
            stock = yf.Ticker(ticker)
            dividends = stock.dividends
            splits = stock.splits

            return {
                "dividends": dividends.to_frame() if not dividends.empty else pd.DataFrame(),
                "splits": splits.to_frame() if not splits.empty else pd.DataFrame(),
            }
        except Exception as e:
            logger.error(f"Error fetching dividends/splits for {ticker}: {e}")
            return {"dividends": pd.DataFrame(), "splits": pd.DataFrame()}

    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        start: str = "2010-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple tickers.

        Returns dict mapping ticker -> DataFrame.
        """
        results = {}
        for ticker in tickers:
            df = self.fetch_ohlcv(ticker, start=start, end=end, interval=interval)
            if not df.empty:
                results[ticker] = df
            else:
                logger.warning(f"Skipping {ticker} - no data available")
        return results

    def fetch_index_components(self, index: str = "sp500") -> List[str]:
        """
        Fetch current components of a major index.

        Args:
            index: One of 'sp500', 'nasdaq100', 'dow30'

        Returns:
            List of ticker symbols
        """
        try:
            if index == "sp500":
                # Use Wikipedia table for S&P 500 components
                table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
                tickers = table[0]["Symbol"].tolist()
                return [t.replace(".", "-") for t in tickers]  # Yahoo uses - instead of .
            elif index == "dow30":
                table = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
                for t in table:
                    if "Symbol" in t.columns:
                        return t["Symbol"].tolist()
            else:
                logger.warning(f"Unknown index: {index}")
                return []
        except Exception as e:
            logger.error(f"Error fetching index components for {index}: {e}")
            return []
