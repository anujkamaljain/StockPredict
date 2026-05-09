"""
FRED (Federal Reserve Economic Data) provider.
Fetches macroeconomic indicators for market regime context.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
from loguru import logger


class FREDProvider:
    """
    Fetches macroeconomic data from FRED API.
    Free with API key. Used for interest rates, inflation, GDP, unemployment, etc.
    """

    # Key macroeconomic series
    MACRO_SERIES = {
        "fed_funds_rate": "FEDFUNDS",
        "treasury_10y": "DGS10",
        "treasury_2y": "DGS2",
        "treasury_3m": "DTB3",
        "yield_spread_10y2y": "T10Y2Y",
        "inflation_cpi": "CPIAUCSL",
        "core_cpi": "CPILFESL",
        "unemployment": "UNRATE",
        "gdp": "GDP",
        "gdp_growth": "A191RL1Q225SBEA",
        "industrial_production": "INDPRO",
        "consumer_sentiment": "UMCSENT",
        "vix": "VIXCLS",
        "dollar_index": "DTWEXBGS",
        "m2_money_supply": "M2SL",
        "housing_starts": "HOUST",
        "retail_sales": "RSXFS",
        "pce": "PCE",
        "initial_claims": "ICSA",
    }

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self.cache_dir = cache_dir
        self._fred = None

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_fred_client(self):
        """Lazy initialization of FRED client."""
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.api_key)
            except ImportError:
                logger.error("fredapi not installed. Install with: pip install fredapi")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize FRED client: {e}")
                return None
        return self._fred

    def _get_cache_path(self, series_id: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"fred_{series_id}.parquet"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        if not cache_path or not cache_path.exists():
            return False
        mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return (datetime.now() - mod_time) < timedelta(hours=max_age_hours)

    def fetch_series(
        self,
        series_id: str,
        start: str = "2005-01-01",
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.Series:
        """
        Fetch a single FRED data series.

        Args:
            series_id: FRED series ID (e.g., 'FEDFUNDS')
            start: Start date
            end: End date
            use_cache: Whether to use cached data

        Returns:
            pandas Series with the data
        """
        cache_path = self._get_cache_path(series_id)

        if use_cache and cache_path and self._is_cache_valid(cache_path):
            logger.info(f"Loading cached FRED data for {series_id}")
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0]

        fred = self._get_fred_client()
        if fred is None:
            logger.warning(f"FRED client not available, returning empty series for {series_id}")
            return pd.Series(dtype=float, name=series_id)

        try:
            logger.info(f"Fetching FRED series: {series_id}")
            data = fred.get_series(series_id, observation_start=start, observation_end=end)

            if data is not None and not data.empty:
                # Cache the data
                if cache_path:
                    data.to_frame().to_parquet(cache_path)
                return data
            else:
                logger.warning(f"No data returned for FRED series {series_id}")
                return pd.Series(dtype=float, name=series_id)

        except Exception as e:
            logger.error(f"Error fetching FRED series {series_id}: {e}")
            return pd.Series(dtype=float, name=series_id)

    def fetch_all_macro_indicators(
        self,
        start: str = "2005-01-01",
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch all macroeconomic indicators and combine into a single DataFrame.
        Forward-fills missing values for alignment with daily stock data.

        Returns:
            DataFrame with macro indicators as columns, DatetimeIndex
        """
        all_data = {}

        for name, series_id in self.MACRO_SERIES.items():
            data = self.fetch_series(series_id, start=start, end=end, use_cache=use_cache)
            if not data.empty:
                all_data[name] = data

        if not all_data:
            logger.warning("No macroeconomic data fetched")
            return pd.DataFrame()

        # Combine all series
        df = pd.DataFrame(all_data)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Forward-fill to align with daily data (macro data is often monthly/quarterly)
        df = df.ffill()

        # Compute derived features
        if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
            df["yield_curve_slope"] = df["treasury_10y"] - df["treasury_2y"]

        if "inflation_cpi" in df.columns:
            df["inflation_yoy"] = df["inflation_cpi"].pct_change(periods=12) * 100

        logger.info(f"Fetched {len(df)} rows of macroeconomic data with {len(df.columns)} indicators")
        return df

    def get_recession_indicators(self, start: str = "2005-01-01") -> pd.DataFrame:
        """
        Compute recession probability indicators.
        Uses yield curve inversion as a primary signal.
        """
        indicators = {}

        # Yield spread (10Y - 2Y)
        spread = self.fetch_series("T10Y2Y", start=start)
        if not spread.empty:
            indicators["yield_spread"] = spread
            indicators["yield_curve_inverted"] = (spread < 0).astype(int)

        # Sahm Rule recession indicator
        sahm = self.fetch_series("SAHMREALTIME", start=start)
        if not sahm.empty:
            indicators["sahm_indicator"] = sahm

        if not indicators:
            return pd.DataFrame()

        df = pd.DataFrame(indicators)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = df.ffill()

        return df
