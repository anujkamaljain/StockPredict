"""
Data API routes — fetch and manage market data.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from loguru import logger
import numpy as np
import pandas as pd

from app.data.ingestion import DataIngestionService
from app.config import config

router = APIRouter()


def _safe_float(x, default=0.0):
    """Coerce to float, replacing NaN/Inf with `default`. Default may be None."""
    try:
        f = float(x)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _safe_int(x, default: int = 0) -> int:
    """Coerce to int, replacing NaN/Inf with default."""
    try:
        f = float(x)
        return int(f) if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


@router.get("/fetch/{ticker}")
async def fetch_stock_data(
    ticker: str,
    start: str = Query("2015-01-01", description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """Fetch OHLCV data for a single stock."""
    try:
        import asyncio
        service = DataIngestionService()
        result = await asyncio.to_thread(
            service.fetch_stock_data, ticker.upper(), start, end,
        )

        if result["ohlcv"].empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        df = result["ohlcv"].copy()

        # Defensive: replace NaN/Inf in numeric columns before JSON serialization.
        # `int(NaN)` would raise IntCastingNaNError, breaking the whole endpoint.
        for col in ("Open", "High", "Low", "Close"):
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "Volume" in df.columns:
            df["Volume"] = (
                df["Volume"].replace([np.inf, -np.inf], np.nan).fillna(0).astype("int64")
            )

        last_pct = df["Close"].pct_change().iloc[-1] if len(df) > 1 else 0.0
        return {
            "ticker": ticker.upper(),
            "source": result["source"],
            "rows": len(df),
            "start": df.index.min().strftime("%Y-%m-%d"),
            "end": df.index.max().strftime("%Y-%m-%d"),
            "data": {
                "dates": [d.strftime("%Y-%m-%d") for d in df.index],
                "open": df["Open"].round(2).tolist(),
                "high": df["High"].round(2).tolist(),
                "low": df["Low"].round(2).tolist(),
                "close": df["Close"].round(2).tolist(),
                "volume": df["Volume"].tolist(),
            },
            "latest": {
                "close": round(_safe_float(df["Close"].iloc[-1]), 2),
                "change": round(_safe_float(last_pct) * 100, 2),
                "volume": _safe_int(df["Volume"].iloc[-1]),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fundamentals/{ticker}")
async def get_fundamentals(ticker: str):
    """Fetch fundamental data for a stock."""
    try:
        service = DataIngestionService()
        fundamentals = service.yahoo.fetch_fundamentals(ticker.upper())
        return fundamentals
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_stocks(q: str = Query(..., min_length=1, description="Search query")):
    """
    Search for stocks by ticker or company name.

    Priority:
    🥇 Alpha Vantage SYMBOL_SEARCH (official API, match scores)
    🥈 Yahoo Finance search (fallback — no API key needed)
    """
    import httpx
    from app.data.providers.alpha_vantage import AlphaVantageProvider

    # --- 🥇 Try Alpha Vantage SYMBOL_SEARCH first ---
    av = AlphaVantageProvider(api_key=config.api_keys.alpha_vantage)
    if av.is_configured:
        try:
            av_results = await av.search_symbols(q)
            if av_results:
                logger.info(f"Alpha Vantage search for '{q}': {len(av_results)} results")
                return av_results
        except Exception as e:
            logger.warning(f"Alpha Vantage search failed for '{q}': {e}")

    # --- 🥈 Fallback to Yahoo Finance search ---
    logger.info(f"Falling back to Yahoo Finance search for '{q}'")
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {
        "q": q,
        "quotesCount": 8,
        "newsCount": 0,
        "listsCount": 0,
        "enableFuzzyQuery": True,
        "quotesQueryId": "tss_match_phrase_query",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        quotes = data.get("quotes", [])
        results = []
        for quote in quotes:
            q_type = quote.get("quoteType", "")
            # Only include equities, ETFs, and indices
            if q_type not in ("EQUITY", "ETF", "INDEX", "MUTUALFUND"):
                continue
            results.append({
                "ticker": quote.get("symbol", ""),
                "name": quote.get("longname") or quote.get("shortname", ""),
                "exchange": quote.get("exchDisp", quote.get("exchange", "")),
                "type": q_type,
                "sector": quote.get("sector", ""),
                "industry": quote.get("industry", ""),
            })
        return results
    except Exception as e:
        logger.warning(f"Yahoo search also failed for '{q}': {e}")
        return []


@router.get("/summary")
async def data_summary():
    """Get summary of all cached data."""
    try:
        service = DataIngestionService()
        return service.get_data_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/macro")
async def get_macro_data(start: str = "2015-01-01"):
    """Fetch macroeconomic indicators."""
    try:
        service = DataIngestionService()
        df = service.fetch_macro_data(start=start)
        if df.empty:
            return {"message": "No macro data available. Set FRED_API_KEY in .env"}

        latest = {}
        for col in df.columns:
            val = _safe_float(df[col].iloc[-1], default=None)  # type: ignore[arg-type]
            if val is not None:
                latest[col] = round(val, 4)
        return {
            "rows": len(df),
            "indicators": list(df.columns),
            "latest": latest,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
