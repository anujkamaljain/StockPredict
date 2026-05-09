"""
Backtesting engine with realistic market simulation.
Includes transaction costs, slippage, position sizing, and capital constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any
from loguru import logger
from dataclasses import dataclass, field


@dataclass
class Trade:
    """Record of a single trade."""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    direction: str  # "LONG" or "SHORT"
    pnl: float
    return_pct: float
    holding_days: int
    exit_reason: str  # "SIGNAL", "STOP_LOSS", "TRAILING_STOP"


class BacktestEngine:
    """
    Realistic backtesting simulator.

    Simulates:
    - Transaction costs (commission + spread)
    - Slippage (market impact)
    - Position sizing constraints
    - Capital limitations
    - Stop-loss execution
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        max_position_pct: float = 0.10,
        stop_loss_pct: float = 0.05,
        risk_free_rate: float = 0.05,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        prices: pd.Series,
        signals: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        ticker: str = "STOCK",
    ) -> Dict[str, Any]:
        """
        Run backtest simulation.

        Args:
            prices: Close prices (aligned with signals)
            signals: Array of signals: 1 (buy), -1 (sell), 0 (hold)
            confidence: Confidence scores for position sizing
            dates: Date index
            ticker: Stock symbol

        Returns:
            Dict with equity curve, trades, and performance metrics
        """
        n = len(prices)
        if dates is None:
            dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

        prices = np.array(prices, dtype=float)
        signals = np.array(signals, dtype=int)

        if confidence is None:
            confidence = np.ones(n) * 0.5

        # State tracking
        capital = self.initial_capital
        shares = 0.0
        position = 0  # 0 = flat, 1 = long
        entry_price = 0.0
        stop_loss_price = 0.0

        # Results
        equity_curve = np.zeros(n)
        trades: List[Trade] = []
        entry_date_idx = 0

        for i in range(n):
            current_price = prices[i]

            # Check stop-loss
            if position == 1 and current_price <= stop_loss_price:
                # Execute stop-loss
                exit_price = current_price * (1 - self.slippage_pct)
                proceeds = shares * exit_price
                cost = proceeds * self.transaction_cost_pct
                pnl = proceeds - cost - (shares * entry_price)
                capital += proceeds - cost

                trades.append(Trade(
                    ticker=ticker,
                    entry_date=str(dates[entry_date_idx]),
                    exit_date=str(dates[i]),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    shares=shares,
                    direction="LONG",
                    pnl=pnl,
                    return_pct=(exit_price / entry_price - 1),
                    holding_days=i - entry_date_idx,
                    exit_reason="STOP_LOSS",
                ))

                shares = 0
                position = 0

            # Process signals
            if signals[i] == 1 and position == 0:
                # BUY
                position_size = capital * self.max_position_pct * confidence[i]
                buy_price = current_price * (1 + self.slippage_pct)
                shares = int(position_size / buy_price)

                if shares > 0:
                    cost = shares * buy_price * (1 + self.transaction_cost_pct)
                    capital -= cost
                    entry_price = buy_price
                    stop_loss_price = buy_price * (1 - self.stop_loss_pct)
                    entry_date_idx = i
                    position = 1

            elif signals[i] == -1 and position == 1:
                # SELL
                exit_price = current_price * (1 - self.slippage_pct)
                proceeds = shares * exit_price
                cost = proceeds * self.transaction_cost_pct
                pnl = proceeds - cost - (shares * entry_price)
                capital += proceeds - cost

                trades.append(Trade(
                    ticker=ticker,
                    entry_date=str(dates[entry_date_idx]),
                    exit_date=str(dates[i]),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    shares=shares,
                    direction="LONG",
                    pnl=pnl,
                    return_pct=(exit_price / entry_price - 1),
                    holding_days=i - entry_date_idx,
                    exit_reason="SIGNAL",
                ))

                shares = 0
                position = 0

            # Track equity
            equity_curve[i] = capital + (shares * current_price if position == 1 else 0)

        # Close any open position
        if position == 1:
            exit_price = prices[-1] * (1 - self.slippage_pct)
            proceeds = shares * exit_price
            capital += proceeds * (1 - self.transaction_cost_pct)

        # Compute metrics
        equity_series = pd.Series(equity_curve, index=dates)
        metrics = self._compute_metrics(equity_series, trades)

        # Buy and hold comparison
        bh_shares = int(self.initial_capital / prices[0])
        bh_equity = bh_shares * prices
        bh_series = pd.Series(bh_equity, index=dates)
        bh_metrics = self._compute_metrics(bh_series, [])

        return {
            "equity_curve": equity_series.to_dict(),
            "trades": [self._trade_to_dict(t) for t in trades],
            "metrics": metrics,
            "buy_hold_metrics": bh_metrics,
            "initial_capital": self.initial_capital,
            "final_capital": equity_curve[-1] if len(equity_curve) > 0 else self.initial_capital,
        }

    def _compute_metrics(
        self, equity: pd.Series, trades: List[Trade]
    ) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        if len(equity) < 2:
            return {}

        returns = equity.pct_change().dropna()
        total_days = (equity.index[-1] - equity.index[0]).days if hasattr(equity.index[0], 'days') else len(equity)
        years = max(total_days / 365.25, 0.01) if isinstance(total_days, (int, float)) else len(equity) / 252

        # Total return
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

        # CAGR
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(years, 0.01)) - 1

        # Volatility
        annual_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0

        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe = (excess_returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252) if len(returns) > 1 else 0

        # Sortino ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-10
        sortino = (returns.mean() * 252 - self.risk_free_rate) / downside_std

        # Max drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        if trades:
            winning = sum(1 for t in trades if t.pnl > 0)
            win_rate = winning / len(trades)
            avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if winning > 0 else 0
            avg_loss = np.mean([abs(t.pnl) for t in trades if t.pnl <= 0]) if (len(trades) - winning) > 0 else 0
            profit_factor = (avg_win * winning) / (avg_loss * (len(trades) - winning) + 1e-10)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            "total_return": round(float(total_return), 4),
            "cagr": round(float(cagr), 4),
            "annual_volatility": round(float(annual_vol), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "sortino_ratio": round(float(sortino), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "calmar_ratio": round(float(calmar), 4),
            "total_trades": len(trades),
            "win_rate": round(float(win_rate), 4),
            "avg_win": round(float(avg_win), 2),
            "avg_loss": round(float(avg_loss), 2),
            "profit_factor": round(float(profit_factor), 4),
        }

    @staticmethod
    def _trade_to_dict(trade: Trade) -> Dict:
        return {
            "ticker": trade.ticker,
            "entry_date": trade.entry_date,
            "exit_date": trade.exit_date,
            "entry_price": round(trade.entry_price, 2),
            "exit_price": round(trade.exit_price, 2),
            "shares": trade.shares,
            "direction": trade.direction,
            "pnl": round(trade.pnl, 2),
            "return_pct": round(trade.return_pct, 4),
            "holding_days": trade.holding_days,
            "exit_reason": trade.exit_reason,
        }
