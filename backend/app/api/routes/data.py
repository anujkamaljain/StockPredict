"""
Data API routes — fetch and manage market data.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from loguru import logger

from app.data.ingestion import DataIngestionService
from app.config import config

router = APIRouter()


@router.get("/fetch/{ticker}")
async def fetch_stock_data(
    ticker: str,
    start: str = Query("2015-01-01", description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """Fetch OHLCV data for a single stock."""
    try:
        service = DataIngestionService()
        result = service.fetch_stock_data(ticker.upper(), start=start, end=end)

        if result["ohlcv"].empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        df = result["ohlcv"]
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
                "volume": df["Volume"].astype(int).tolist(),
            },
            "latest": {
                "close": round(float(df["Close"].iloc[-1]), 2),
                "change": round(float(df["Close"].pct_change().iloc[-1] * 100), 2),
                "volume": int(df["Volume"].iloc[-1]),
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
    """Search for stocks by ticker or name using Yahoo Finance autocomplete."""
    import aiohttp

    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {
        "q": q,
        "quotesCount": 10,
        "newsCount": 0,
        "listsCount": 0,
        "enableFuzzyQuery": True,
        "quotesQueryId": "tss_match_phrase_query",
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.warning(f"Yahoo search returned {resp.status} for '{q}'")
                    return []
                data = await resp.json()

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
        logger.warning(f"Search failed for '{q}': {e}")
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

        return {
            "rows": len(df),
            "indicators": list(df.columns),
            "latest": {col: round(float(df[col].iloc[-1]), 4) for col in df.columns if not df[col].iloc[-1] != df[col].iloc[-1]},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
