"""
Portfolio API routes — manage user portfolio and risk.
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from loguru import logger

from app.risk.manager import RiskManager

router = APIRouter()

# In-memory portfolio state (would be DB in production)
_risk_managers: Dict[str, RiskManager] = {}


class PortfolioConfig(BaseModel):
    """User portfolio configuration."""
    initial_capital: float = 100000
    max_position_pct: float = 0.10
    stop_loss_pct: float = 0.05
    trailing_stop_pct: float = 0.03
    max_drawdown_limit: float = 0.15
    kelly_fraction: float = 0.5


@router.post("/configure")
async def configure_portfolio(cfg: PortfolioConfig):
    """Configure portfolio risk parameters (dynamic per user)."""
    rm = RiskManager(
        initial_capital=cfg.initial_capital,
        max_position_pct=cfg.max_position_pct,
        stop_loss_pct=cfg.stop_loss_pct,
        trailing_stop_pct=cfg.trailing_stop_pct,
        max_drawdown_limit=cfg.max_drawdown_limit,
        kelly_fraction=cfg.kelly_fraction,
    )
    _risk_managers["default"] = rm
    return {"status": "configured", "config": cfg.model_dump()}


@router.get("/status")
async def portfolio_status():
    """Get current portfolio risk status."""
    rm = _risk_managers.get("default")
    if not rm:
        rm = RiskManager()
        _risk_managers["default"] = rm

    return rm.check_portfolio_risk()


@router.get("/position-size")
async def compute_position_size(
    direction_prob: float = Query(..., ge=0, le=1),
    confidence: float = Query(..., ge=0, le=1),
    current_price: float = Query(..., gt=0),
    expected_return: Optional[float] = None,
    volatility: Optional[float] = None,
):
    """Compute recommended position size using Kelly Criterion."""
    rm = _risk_managers.get("default", RiskManager())

    result = rm.compute_position_size(
        direction_prob=direction_prob,
        confidence=confidence,
        expected_return=expected_return,
        volatility=volatility,
        current_price=current_price,
    )
    return result


@router.get("/stop-loss")
async def compute_stop_loss(
    ticker: str,
    entry_price: float = Query(..., gt=0),
):
    """Compute stop-loss and trailing stop levels."""
    rm = _risk_managers.get("default", RiskManager())
    return rm.set_stop_loss(ticker, entry_price)
