"""
FastAPI main application entry point.

Production-ready:
- CORS allow-list driven by FRONTEND_URL env var (comma-separated supported)
- Startup hook preloads trained models so the first request is fast
- /health endpoint reports model availability for liveness checks
- Logs go to stderr and to a rotating file
"""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import backtest, data, portfolio, signals
from app.config import config
from app.models.inference import get_inferencer


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan> - {message}",
)
log_dir = config.data.data_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(log_dir / "app.log", rotation="10 MB", level="DEBUG")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Stock Market ML Decision Support System",
    description=(
        "Production-grade ML pipeline for stock market analysis with ensemble "
        "models, risk management, and backtesting."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — accept comma-separated list in FRONTEND_URL.
# Always allow localhost for dev convenience.
_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_extra = [u.strip() for u in os.getenv("FRONTEND_URL", "").split(",") if u.strip()]
_origins = list(dict.fromkeys(_default_origins + _extra))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Routers
app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(signals.router, prefix="/api/signals", tags=["Signals"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest"])


# ---------------------------------------------------------------------------
# Startup hook
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _on_startup() -> None:
    logger.info(f"CORS allow-list: {_origins}")
    # Eagerly load trained models so the first user request is fast
    inferencer = get_inferencer()
    info = inferencer.info()
    if info["available"]:
        loaded = ", ".join(info["loaded_models"])
        logger.info(
            f"Trained models loaded: [{loaded}] "
            f"(n_features={info['n_features']}, seq_length={info['seq_length']}, "
            f"gates_passed={info['report'].get('gates_passed')})"
        )
    else:
        logger.warning(
            "No trained models found in data/models/. "
            "Signals will use the statistical fallback. "
            "Train via train_cloud.py and drop trained_models/* into data/models/."
        )


# ---------------------------------------------------------------------------
# Root + health
# ---------------------------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    """Tiny landing payload."""
    return {
        "name": "Stock Market ML Decision Support System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
async def health():
    """
    Liveness probe + model availability summary.

    Used by Cloud Run / Kubernetes / uptime monitors.
    Returns 200 always; the `models_loaded` flag tells you if the ML path is
    live or the API is in statistical-fallback mode.
    """
    info = get_inferencer().info()
    return {
        "status": "healthy",
        "models_loaded": info["available"],
        "loaded_models": info["loaded_models"],
        "gates_passed": info["report"].get("gates_passed"),
    }
