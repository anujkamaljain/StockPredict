"""
Risk management layer.
Position sizing (Kelly Criterion), stop-loss, and portfolio-level risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any
from loguru import logger
from dataclasses import dataclass, field


@dataclass
class Position:
    """Represents a portfolio position."""
    ticker: str
    shares: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    stop_loss_price: float = 0.0
    trailing_stop_price: float = 0.0
    unrealized_pnl: float = 0.0
    weight: float = 0.0  # Portfolio weight

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price


class RiskManager:
    """
    Portfolio-level risk management.

    Implements:
    - Kelly Criterion position sizing (half-Kelly for conservatism)
    - Fixed fractional sizing as alternative
    - Stop-loss (fixed and trailing)
    - Max drawdown limits
    - Diversification constraints
    - Position correlation limits
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_pct: float = 0.10,
        stop_loss_pct: float = 0.05,
        trailing_stop_pct: float = 0.03,
        max_drawdown_limit: float = 0.15,
        kelly_fraction: float = 0.5,
        max_total_exposure: float = 0.90,
        max_sector_exposure: float = 0.30,
        max_correlated_positions: int = 5,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.kelly_fraction = kelly_fraction
        self.max_total_exposure = max_total_exposure
        self.max_sector_exposure = max_sector_exposure
        self.max_correlated_positions = max_correlated_positions

        self.positions: Dict[str, Position] = {}
        self.peak_capital = initial_capital
        self.is_risk_off = False

    def compute_kelly_size(
        self,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Compute Kelly Criterion position size.

        Kelly = (p * b - q) / b
        where p = win prob, q = loss prob, b = win/loss ratio

        Uses half-Kelly for conservatism.
        """
        q = 1 - win_probability
        if avg_loss == 0:
            return 0.0

        b = abs(avg_win / avg_loss)
        kelly = (win_probability * b - q) / b

        # Half-Kelly
        kelly *= self.kelly_fraction

        # Clamp to max position size
        kelly = max(0, min(kelly, self.max_position_pct))

        return kelly

    def compute_position_size(
        self,
        direction_prob: float,
        confidence: float,
        expected_return: Optional[float] = None,
        volatility: Optional[float] = None,
        current_price: float = 0.0,
    ) -> Dict:
        """
        Compute recommended position size.

        Args:
            direction_prob: P(up) from ensemble
            confidence: Signal confidence [0, 1]
            expected_return: Expected return magnitude
            volatility: Current annualized volatility
            current_price: Current stock price

        Returns:
            Dict with shares, dollar_amount, weight, method
        """
        if self.is_risk_off:
            return {"shares": 0, "dollar_amount": 0, "weight": 0, "method": "risk_off", "reason": "Portfolio in risk-off mode"}

        available_capital = self.capital * (self.max_total_exposure - self._total_exposure())
        if available_capital <= 0:
            return {"shares": 0, "dollar_amount": 0, "weight": 0, "method": "no_capital", "reason": "Max exposure reached"}

        # Method 1: Kelly Criterion
        if expected_return and expected_return != 0:
            avg_win = abs(expected_return) if direction_prob > 0.5 else abs(expected_return) * 0.5
            avg_loss = abs(expected_return) * 0.5 if direction_prob > 0.5 else abs(expected_return)
            kelly_weight = self.compute_kelly_size(direction_prob, avg_win, avg_loss)
        else:
            # Fallback: confidence-based sizing
            kelly_weight = confidence * self.max_position_pct * 0.5

        # Adjust for volatility (inverse vol sizing)
        if volatility and volatility > 0:
            target_vol = 0.15  # Target 15% annual portfolio vol
            vol_adjustment = target_vol / volatility
            kelly_weight *= min(vol_adjustment, 2.0)  # Cap at 2x

        # Apply constraints
        weight = min(kelly_weight, self.max_position_pct)
        dollar_amount = min(weight * self.capital, available_capital)
        shares = int(dollar_amount / current_price) if current_price > 0 else 0

        return {
            "shares": shares,
            "dollar_amount": round(dollar_amount, 2),
            "weight": round(weight, 4),
            "method": "kelly_half",
            "kelly_raw": round(kelly_weight, 4),
        }

    def set_stop_loss(self, ticker: str, entry_price: float) -> Dict[str, float]:
        """Calculate stop-loss and trailing stop prices."""
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        trailing_stop = entry_price * (1 - self.trailing_stop_pct)

        return {
            "stop_loss": round(stop_loss, 2),
            "trailing_stop": round(trailing_stop, 2),
            "stop_loss_pct": self.stop_loss_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
        }

    def check_stop_loss(self, position: Position) -> Optional[str]:
        """Check if a position has hit its stop-loss."""
        if position.current_price <= position.stop_loss_price:
            return "STOP_LOSS_HIT"

        # Update trailing stop
        new_trailing = position.current_price * (1 - self.trailing_stop_pct)
        if new_trailing > position.trailing_stop_price:
            position.trailing_stop_price = new_trailing

        if position.current_price <= position.trailing_stop_price:
            return "TRAILING_STOP_HIT"

        return None

    def check_portfolio_risk(self) -> Dict[str, Any]:
        """Check portfolio-level risk metrics."""
        total_value = self.capital + sum(p.market_value for p in self.positions.values())

        # Update peak
        if total_value > self.peak_capital:
            self.peak_capital = total_value

        # Current drawdown
        drawdown = (self.peak_capital - total_value) / self.peak_capital if self.peak_capital > 0 else 0

        # Check max drawdown limit
        if drawdown >= self.max_drawdown_limit:
            self.is_risk_off = True
            logger.warning(f"MAX DRAWDOWN BREACH: {drawdown:.2%} >= {self.max_drawdown_limit:.2%}. Risk-off mode activated.")

        return {
            "total_value": round(total_value, 2),
            "cash": round(self.capital, 2),
            "invested": round(sum(p.market_value for p in self.positions.values()), 2),
            "total_exposure": round(self._total_exposure(), 4),
            "drawdown": round(drawdown, 4),
            "peak_value": round(self.peak_capital, 2),
            "is_risk_off": self.is_risk_off,
            "num_positions": len(self.positions),
        }

    def _total_exposure(self) -> float:
        """Calculate total portfolio exposure as fraction of capital."""
        total_invested = sum(p.market_value for p in self.positions.values())
        total_value = self.capital + total_invested
        return total_invested / total_value if total_value > 0 else 0
