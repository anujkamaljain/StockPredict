# Stock Alert System — Implementation Plan

> **Goal:** Automatically scan your watchlist stocks daily after market close,
> run them through the ML ensemble, and notify you via **Telegram + Email**
> when a high-confidence BUY or SELL signal appears.

---

## Architecture

```
┌──────────────┐    daily hit     ┌──────────────────────┐
│ cronjob.org  │────────────────▶│ FastAPI Backend       │
│ (scheduler)  │   GET /api/     │                      │
│              │   alerts/scan   │  1. Load watchlist    │
└──────────────┘   + secret key  │  2. Fetch latest data │
                                 │  3. Run ML ensemble   │
                                 │  4. Filter high-conf  │
                                 │  5. Send notifications│
                                 └──────────┬───────────┘
                                            │
                                    ┌───────┴───────┐
                                    ▼               ▼
                          ┌────────────────┐ ┌──────────────┐
                          │ Telegram Bot   │ │ Resend Email │
                          │ → phone push   │ │ → inbox      │
                          └────────────────┘ └──────────────┘
```

### Why this design?

| Decision | Reason |
|----------|--------|
| **cronjob.org** (not local cron) | Works even if your laptop is off; free; no infra |
| **Telegram** | Instant push notification, free, no spam folder |
| **Resend email** | Free 100 emails/day; permanent record in inbox; backup if Telegram fails |
| **Secret key on endpoint** | Prevents random people from triggering your scans |
| **Server-side watchlist** | Persists across sessions; editable from frontend |

---

## What gets built

### Backend (3 new files + 1 modified)

```
backend/
├── app/
│   ├── alerts/
│   │   ├── __init__.py
│   │   ├── watchlist.py        # Watchlist CRUD (JSON file storage)
│   │   ├── scanner.py          # Scan all watchlist stocks through ML
│   │   └── notifier.py         # Telegram notification sender
│   ├── api/routes/
│   │   └── alerts.py           # New API routes
│   └── main.py                 # Register new router (modify)
├── data/
│   └── watchlist.json          # Persistent watchlist storage
└── .env                        # Add TELEGRAM_*, RESEND_*, ALERT_SECRET
```

### Frontend (1 new page + header link)

```
frontend/src/
├── app/watchlist/
│   └── page.tsx                # Watchlist management UI
└── components/
    └── WatchlistButton.tsx     # Quick add-to-watchlist from stock page
```

---

## Step-by-step implementation

### Phase 1A: Telegram Bot Setup (5 minutes)

1. **Create the bot:**
   - Open Telegram → search `@BotFather` → send `/newbot`
   - Name it: `StockML Alert Bot`
   - Username: `stockml_yourname_bot`
   - BotFather gives you a **token** like `7123456789:AAH...` → save it

2. **Get your chat ID:**
   - Search your new bot in Telegram → send it any message (e.g., "hello")
   - Open this URL in browser:
     ```
     https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
     ```
   - Find `"chat":{"id": 123456789}` → that's your **chat ID**

### Phase 1B: Resend Email Setup (3 minutes)

1. Go to [https://resend.com](https://resend.com) → sign up (free)
2. Free tier gives you **100 emails/day** and **3,000 emails/month** — more than enough
3. Go to **API Keys** → click **Create API Key** → copy the key (starts with `re_...`)
4. Note: on free tier, you can only send **to your own email** (the one you signed up with) — perfect for personal alerts

### Add all credentials to `.env`:

```env
# Alert System
TELEGRAM_BOT_TOKEN=7123456789:AAHxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=123456789
RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALERT_EMAIL_TO=your.email@gmail.com
ALERT_SECRET=any-random-secret-string-here
```

### Phase 2: Backend — Watchlist Storage

**`backend/app/alerts/watchlist.py`**

```python
"""
Watchlist: simple JSON file-based storage for watched tickers.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

WATCHLIST_PATH = Path(__file__).parent.parent.parent / "data" / "watchlist.json"


def _load() -> Dict:
    if WATCHLIST_PATH.exists():
        return json.loads(WATCHLIST_PATH.read_text())
    return {"stocks": [], "updated_at": None}


def _save(data: Dict):
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat()
    WATCHLIST_PATH.write_text(json.dumps(data, indent=2))


def get_watchlist() -> List[Dict]:
    return _load()["stocks"]


def add_ticker(ticker: str, notes: str = "") -> bool:
    data = _load()
    # Check if already exists
    if any(s["ticker"] == ticker.upper() for s in data["stocks"]):
        return False
    data["stocks"].append({
        "ticker": ticker.upper(),
        "added_at": datetime.now().isoformat(),
        "notes": notes,
        "min_confidence": 0.6,  # default alert threshold
    })
    _save(data)
    return True


def remove_ticker(ticker: str) -> bool:
    data = _load()
    before = len(data["stocks"])
    data["stocks"] = [s for s in data["stocks"] if s["ticker"] != ticker.upper()]
    if len(data["stocks"]) < before:
        _save(data)
        return True
    return False


def update_threshold(ticker: str, min_confidence: float) -> bool:
    data = _load()
    for s in data["stocks"]:
        if s["ticker"] == ticker.upper():
            s["min_confidence"] = min_confidence
            _save(data)
            return True
    return False
```

### Phase 3: Backend — Notifier (Telegram + Email)

**`backend/app/alerts/notifier.py`**

```python
"""
Send alert notifications via Telegram Bot API and Resend Email.
Both channels fire independently — if one fails, the other still works.
"""
import os
import httpx
from typing import List, Dict
from loguru import logger


def _build_text_message(signals: List[Dict]) -> str:
    """Build plain-text message for Telegram."""
    lines = ["🚨 Stock Alert — ML Signal Triggered\n"]
    for s in signals:
        emoji = "🟢" if s["action"] == "BUY" else "🔴"
        lines.append(
            f"{emoji} {s['ticker']} → {s['action']}\n"
            f"   Confidence: {s['confidence']:.0%}\n"
            f"   P(up): {s['direction_prob']:.2%}\n"
            f"   Price: {s['price']}\n"
            f"   Model agreement: {s['model_agreement']:.0%}\n"
        )
    lines.append(f"\nScanned {signals[0].get('total_scanned', '?')} watchlist stocks")
    return "\n".join(lines)


def _build_html_email(signals: List[Dict]) -> str:
    """Build HTML email body for Resend."""
    rows = ""
    for s in signals:
        color = "#22c55e" if s["action"] == "BUY" else "#ef4444"
        rows += f"""
        <tr>
            <td style="padding:8px;font-weight:bold">{s['ticker']}</td>
            <td style="padding:8px;color:{color};font-weight:bold">{s['action']}</td>
            <td style="padding:8px">{s['confidence']:.0%}</td>
            <td style="padding:8px">{s['direction_prob']:.2%}</td>
            <td style="padding:8px">{s['price']}</td>
            <td style="padding:8px">{s['model_agreement']:.0%}</td>
        </tr>"""

    return f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto">
        <h2 style="color:#1e293b">🚨 Stock Alert — ML Signal Triggered</h2>
        <table style="width:100%;border-collapse:collapse;border:1px solid #e2e8f0">
            <thead>
                <tr style="background:#f8fafc">
                    <th style="padding:8px;text-align:left">Ticker</th>
                    <th style="padding:8px;text-align:left">Action</th>
                    <th style="padding:8px;text-align:left">Confidence</th>
                    <th style="padding:8px;text-align:left">P(up)</th>
                    <th style="padding:8px;text-align:left">Price</th>
                    <th style="padding:8px;text-align:left">Agreement</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        <p style="color:#64748b;font-size:12px;margin-top:16px">
            Scanned {signals[0].get('total_scanned', '?')} watchlist stocks
        </p>
    </div>
    """


def send_telegram_alert(signals: List[Dict]) -> bool:
    """Send alert via Telegram Bot API."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        logger.warning("Telegram credentials not set, skipping")
        return False

    message = _build_text_message(signals)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = httpx.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
        if resp.status_code == 200:
            logger.info(f"Telegram alert sent: {len(signals)} signals")
            return True
        else:
            logger.error(f"Telegram API error: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def send_email_alert(signals: List[Dict]) -> bool:
    """Send alert via Resend email API (free tier: 100/day)."""
    api_key = os.getenv("RESEND_API_KEY", "")
    to_email = os.getenv("ALERT_EMAIL_TO", "")

    if not api_key or not to_email:
        logger.warning("Resend credentials not set, skipping email")
        return False

    # Count actions for subject line
    buys = sum(1 for s in signals if s["action"] == "BUY")
    sells = sum(1 for s in signals if s["action"] == "SELL")
    subject = f"🚨 Stock Alert: {buys} BUY, {sells} SELL signals triggered"

    html_body = _build_html_email(signals)

    try:
        resp = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "from": "Stock Alerts <onboarding@resend.dev>",
                "to": [to_email],
                "subject": subject,
                "html": html_body,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info(f"Email alert sent to {to_email}: {len(signals)} signals")
            return True
        else:
            logger.error(f"Resend API error: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Email send failed: {e}")
        return False


def send_all_alerts(signals: List[Dict]) -> Dict[str, bool]:
    """Send alerts via all configured channels."""
    return {
        "telegram": send_telegram_alert(signals),
        "email": send_email_alert(signals),
    }
```

### Phase 4: Backend — Scanner

**`backend/app/alerts/scanner.py`**

```python
"""
Scan watchlist stocks through the ML ensemble and trigger alerts.
"""
from typing import List, Dict
from loguru import logger

from app.alerts.watchlist import get_watchlist
from app.data.ingestion import DataIngestionService
from app.features.engineering import FeatureEngineer
from app.signals.generator import SignalGenerator
from app.models.inference import get_inferencer


def scan_watchlist() -> Dict:
    """
    Scan all watchlisted stocks, run ML inference, and
    send Telegram alerts for high-confidence signals.

    Returns summary of scan results.
    """
    watchlist = get_watchlist()
    if not watchlist:
        return {"scanned": 0, "alerts_sent": 0, "message": "Watchlist is empty"}

    service = DataIngestionService()
    engineer = FeatureEngineer()
    signal_gen = SignalGenerator(risk_tolerance="medium")
    inferencer = get_inferencer()

    triggered = []
    scanned = 0
    errors = []

    for item in watchlist:
        ticker = item["ticker"]
        min_conf = item.get("min_confidence", 0.6)

        try:
            result = service.fetch_stock_data(ticker, "2015-01-01")
            if result["ohlcv"].empty:
                errors.append(f"{ticker}: no data")
                continue

            df = result["ohlcv"]
            featured_df = engineer.compute_features(df)

            # Same NaN handling as the signal endpoint
            core_cols = [c for c in ["Close", "simple_return", "rsi_14", "macd", "sma_50"]
                         if c in featured_df.columns]
            if core_cols:
                featured_df = featured_df.dropna(subset=core_cols)
            featured_df = featured_df.fillna(0)

            import numpy as np
            featured_df = featured_df.replace([np.inf, -np.inf], 0)

            if len(featured_df) < 50:
                errors.append(f"{ticker}: insufficient data ({len(featured_df)} rows)")
                continue

            # Run ML inference
            ml_result = inferencer.predict_latest(featured_df)

            if ml_result is not None:
                proba = ml_result["ensemble_proba"]
                individual = ml_result["individual_predictions"]
            else:
                continue  # Skip if no ML models loaded

            signal = signal_gen.generate_signal(
                ticker=ticker,
                ensemble_proba=proba,
                individual_predictions=individual,
            )

            scanned += 1

            # Check if signal meets alert threshold
            if signal.confidence >= min_conf and signal.action in ("BUY", "SELL"):
                triggered.append({
                    "ticker": ticker,
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "direction_prob": signal.direction_prob,
                    "model_agreement": signal.model_agreement,
                    "price": round(float(df["Close"].iloc[-1]), 2),
                    "total_scanned": len(watchlist),
                })

        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
            logger.error(f"Alert scan error for {ticker}: {e}")

    # Send notifications via all channels
    alerts_sent = 0
    channel_results = {}
    if triggered:
        from app.alerts.notifier import send_all_alerts
        channel_results = send_all_alerts(triggered)
        alerts_sent = len(triggered) if any(channel_results.values()) else 0

    return {
        "scanned": scanned,
        "total_watchlist": len(watchlist),
        "alerts_triggered": len(triggered),
        "alerts_sent": alerts_sent,
        "channels": channel_results,
        "triggered": triggered,
        "errors": errors,
    }
```

### Phase 5: Backend — API Routes

**`backend/app/api/routes/alerts.py`**

```python
"""
Alert system API routes — watchlist management and scan trigger.
"""
import os
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from app.alerts.watchlist import (
    get_watchlist, add_ticker, remove_ticker, update_threshold
)
from app.alerts.scanner import scan_watchlist

router = APIRouter()


class AddTickerRequest(BaseModel):
    ticker: str
    notes: Optional[str] = ""
    min_confidence: Optional[float] = 0.6


class UpdateThresholdRequest(BaseModel):
    ticker: str
    min_confidence: float


# ---- Watchlist CRUD ----

@router.get("/watchlist")
async def list_watchlist():
    """Get all watchlisted stocks."""
    return {"watchlist": get_watchlist()}


@router.post("/watchlist")
async def add_to_watchlist(req: AddTickerRequest):
    """Add a stock to the watchlist."""
    added = add_ticker(req.ticker, req.notes)
    if not added:
        raise HTTPException(400, f"{req.ticker} is already in watchlist")
    if req.min_confidence != 0.6:
        update_threshold(req.ticker, req.min_confidence)
    return {"added": req.ticker.upper(), "watchlist": get_watchlist()}


@router.delete("/watchlist/{ticker}")
async def remove_from_watchlist(ticker: str):
    """Remove a stock from the watchlist."""
    removed = remove_ticker(ticker)
    if not removed:
        raise HTTPException(404, f"{ticker} not found in watchlist")
    return {"removed": ticker.upper(), "watchlist": get_watchlist()}


@router.put("/watchlist/threshold")
async def set_threshold(req: UpdateThresholdRequest):
    """Update the alert confidence threshold for a stock."""
    updated = update_threshold(req.ticker, req.min_confidence)
    if not updated:
        raise HTTPException(404, f"{req.ticker} not found in watchlist")
    return {"updated": req.ticker.upper(), "min_confidence": req.min_confidence}


# ---- Scan trigger ----

@router.get("/scan")
async def trigger_scan(
    secret: str = Query(..., description="Secret key to authorize scan"),
):
    """
    Scan all watchlist stocks through the ML model.
    Sends Telegram alerts for high-confidence signals.

    This endpoint is meant to be called by cronjob.org daily.
    Requires ?secret=YOUR_ALERT_SECRET for authorization.
    """
    expected = os.getenv("ALERT_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(403, "Invalid secret key")

    import asyncio
    result = await asyncio.to_thread(scan_watchlist)
    return result
```

### Phase 6: Register the router

**In `backend/app/main.py`**, add:

```python
from app.api.routes.alerts import router as alerts_router
app.include_router(alerts_router, prefix="/api/alerts", tags=["alerts"])
```

---

## cronjob.org Setup

1. Go to [https://console.cron-job.org](https://console.cron-job.org) → create free account
2. Click **"Create cronjob"**
3. Configure:

| Field | Value |
|-------|-------|
| **Title** | Stock ML Daily Scan |
| **URL** | `https://your-domain.com/api/alerts/scan?secret=YOUR_ALERT_SECRET` |
| **Schedule** | Custom: specific times |
| **Time** | **16:30 IST** (Indian market close) AND **21:30 IST** (US market close) |
| **Days** | Monday – Friday only |
| **Request method** | GET |
| **Timeout** | 120 seconds |

> **Important:** Your backend must be publicly accessible (deployed, not localhost).
> If running locally only, use a tunnel like `ngrok` or `cloudflared` instead.

### Schedule explanation:
- **16:30 IST** → NSE/BSE closes at 15:30, data settles by 16:30
- **21:30 IST** → NYSE closes at 21:00 IST, data settles by 21:30
- Weekend: no scan needed (markets closed)

---

## What the notifications look like

### Telegram (push notification on phone):

```
🚨 Stock Alert — ML Signal Triggered

🟢 RELIANCE.NS → BUY
   Confidence: 78%
   P(up): 72.3%
   Price: 2847.50
   Model agreement: 83%

🔴 SNAP → SELL
   Confidence: 71%
   P(up): 28.1%
   Price: 11.42
   Model agreement: 67%

Scanned 25 watchlist stocks
```

### Email (formatted HTML table in your inbox):

```
Subject: 🚨 Stock Alert: 1 BUY, 1 SELL signals triggered

┌──────────┬────────┬────────────┬───────┬─────────┬───────────┐
│ Ticker   │ Action │ Confidence │ P(up) │ Price   │ Agreement │
├──────────┼────────┼────────────┼───────┼─────────┼───────────┤
│ RELIANCE │ BUY    │ 78%        │ 72.3% │ 2847.50 │ 83%       │
│ SNAP     │ SELL   │ 71%        │ 28.1% │ 11.42   │ 67%       │
└──────────┴────────┴────────────┴───────┴─────────┴───────────┘
```

---

## Frontend Watchlist UI (optional, Phase 2)

A simple page at `/watchlist` with:
- **Add ticker** input field with search
- **Table** of watchlisted stocks with:
  - Ticker name
  - Current price
  - Last signal (BUY/SELL/HOLD)
  - Alert threshold slider (50%–90%)
  - Remove button
- **"Scan now"** button for manual trigger

---

## Environment variables summary

Add these to `backend/.env`:

```env
# Alert System — Telegram
TELEGRAM_BOT_TOKEN=7123456789:AAHxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=123456789

# Alert System — Email (Resend, free tier: 100/day)
RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALERT_EMAIL_TO=your.email@gmail.com

# Alert System — Security
ALERT_SECRET=my-super-secret-key-12345
```

> **Note:** Both channels are optional. If you only set Telegram creds, only Telegram fires.
> If you only set Resend creds, only email fires. Both set = both fire.

---

## Build order

| Step | What | Time |
|------|------|------|
| 1 | Create Telegram bot, get token + chat ID | 5 min |
| 2 | Create Resend account, get API key | 3 min |
| 3 | Add `.env` variables | 2 min |
| 4 | Build backend files (watchlist, scanner, notifier, routes) | 30 min |
| 5 | Register router in main.py | 2 min |
| 6 | Test locally: `curl localhost:8000/api/alerts/scan?secret=xxx` | 5 min |
| 7 | Deploy backend (so it's publicly accessible) | varies |
| 8 | Set up cronjob.org | 5 min |
| 9 | (Optional) Build frontend watchlist page | 30 min |

**Total: ~1–2 hours of work, after deployment is done.**

---

## Prerequisites

> [!IMPORTANT]
> Build this AFTER:
> 1. ✅ Model training completes on DGX
> 2. ✅ Models are deployed locally and working
> 3. ✅ Backend is deployed to a public URL (for cronjob.org to reach it)
