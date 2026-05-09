"""
Alpha Vantage data provider — PRIMARY source for market data and additional indicators.
Provides access to OHLCV, technical indicators, fundamental data, symbol search,
and economic indicators.

NOTE: Free tier = 25 requests/day, 5/min. Rate limiting is enforced.
"""

import os
import time
import json
import pandas as pd
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
from loguru import logger


class AlphaVantageProvider:
    """
    Primary data source for the StockML system.
    Fetches data from the Alpha Vantage API.

    Provides:
    - Daily OHLCV data (TIME_SERIES_DAILY)
    - Company overview / fundamentals (OVERVIEW)
    - Income statements (INCOME_STATEMENT)
    - Earnings data (EARNINGS)
    - Symbol search (SYMBOL_SEARCH)

    Free tier: 25 requests/day, 5 requests/minute.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.cache_dir = cache_dir
        self._request_count = 0
        self._last_request_time = 0.0
        self._rate_limit_delay = 12.0  # seconds between requests (free tier: 5/min)

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning("Alpha Vantage API key not set. Limited functionality available.")

    @property
    def is_configured(self) -> bool:
        """Check if a valid API key is set."""
        return bool(self.api_key) and self.api_key not in ("", "your_alpha_vantage_key_here")

    def _rate_limit(self):
        """Enforce rate limiting for free tier."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            sleep_time = self._rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
        self._request_count += 1

    def _get_cache_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"av_{key}.json"

    def _read_cache(self, key: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Read from cache if valid."""
        cache_path = self._get_cache_path(key)
        if not cache_path or not cache_path.exists():
            return None
        mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - mod_time) > timedelta(hours=max_age_hours):
            return None
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, key: str, data: Dict):
        """Write data to cache."""
        cache_path = self._get_cache_path(key)
        if cache_path:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

    def _sync_request(self, params: Dict[str, str]) -> Optional[Dict]:
        """Make a synchronous API request."""
        import requests

        if not self.api_key:
            logger.error("Alpha Vantage API key not configured")
            return None

        params["apikey"] = self.api_key
        self._rate_limit()

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None
            if "Note" in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return None

            return data
        except Exception as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            return None

    async def _async_request(self, params: Dict[str, str]) -> Optional[Dict]:
        """Make an async API request (for search, etc.)."""
        import httpx

        if not self.api_key:
            return None

        params["apikey"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None
            if "Note" in data:
                logger.warning(f"Alpha Vantage rate limit hit")
                return None

            return data
        except Exception as e:
            logger.error(f"Alpha Vantage async request failed: {e}")
            return None

    def fetch_daily(
        self,
        ticker: str,
        output_size: str = "full",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data.

        Args:
            ticker: Stock symbol
            output_size: 'compact' (100 days) or 'full' (20+ years)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{ticker}_daily_{output_size}"
        if use_cache:
            cached = self._read_cache(cache_key)
            if cached:
                logger.info(f"Using cached Alpha Vantage data for {ticker}")
                df = pd.DataFrame.from_dict(cached, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df.sort_index(inplace=True)
                return df

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": output_size,
        }

        data = self._sync_request(params)
        if not data or "Time Series (Daily)" not in data:
            return pd.DataFrame()

        ts_data = data["Time Series (Daily)"]
        self._write_cache(cache_key, ts_data)

        df = pd.DataFrame.from_dict(ts_data, orient='index')
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df.sort_index(inplace=True)

        logger.info(f"Fetched {len(df)} daily records for {ticker} from Alpha Vantage")
        return df

    async def search_symbols(self, keywords: str, use_cache: bool = True) -> List[Dict[str, str]]:
        """
        Search for stock symbols using Alpha Vantage SYMBOL_SEARCH.

        Args:
            keywords: Search keywords (ticker or company name)
            use_cache: Whether to use cached results

        Returns:
            List of matching symbols with metadata
        """
        cache_key = f"search_{keywords.upper().replace(' ', '_')}"
        if use_cache:
            cached = self._read_cache(cache_key, max_age_hours=168)  # 1 week cache for search
            if cached:
                logger.debug(f"Using cached search results for '{keywords}'")
                return cached

        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
        }

        data = await self._async_request(params)
        if not data or "bestMatches" not in data:
            return []

        results = []
        for match in data["bestMatches"]:
            results.append({
                "ticker": match.get("1. symbol", ""),
                "name": match.get("2. name", ""),
                "type": match.get("3. type", "Equity"),
                "region": match.get("4. region", ""),
                "exchange": match.get("8. currency", "") or match.get("4. region", ""),
                "currency": match.get("8. currency", "USD"),
                "match_score": match.get("9. matchScore", "0"),
            })

        if results:
            self._write_cache(cache_key, results)
            logger.info(f"Alpha Vantage search for '{keywords}': {len(results)} results")

        return results

    def fetch_company_overview(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        """Fetch company fundamentals overview."""
        cache_key = f"{ticker}_overview"
        if use_cache:
            cached = self._read_cache(cache_key, max_age_hours=168)  # 1 week cache
            if cached:
                return cached

        params = {
            "function": "OVERVIEW",
            "symbol": ticker,
        }

        data = self._sync_request(params)
        if data:
            self._write_cache(cache_key, data)
        return data or {}

    def fetch_income_statement(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        """Fetch income statement data."""
        cache_key = f"{ticker}_income"
        if use_cache:
            cached = self._read_cache(cache_key, max_age_hours=168)
            if cached:
                return cached

        params = {
            "function": "INCOME_STATEMENT",
            "symbol": ticker,
        }

        data = self._sync_request(params)
        if data:
            self._write_cache(cache_key, data)
        return data or {}

    def fetch_earnings(self, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
        """Fetch earnings data."""
        cache_key = f"{ticker}_earnings"
        if use_cache:
            cached = self._read_cache(cache_key, max_age_hours=168)
            if cached:
                return cached

        params = {
            "function": "EARNINGS",
            "symbol": ticker,
        }

        data = self._sync_request(params)
        if data:
            self._write_cache(cache_key, data)
        return data or {}

    @property
    def requests_remaining(self) -> int:
        """Estimate remaining API requests for today (free tier: 25/day)."""
        return max(0, 25 - self._request_count)
