"""
Signal generator: converts model outputs into actionable trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Signal:
    """A single trading signal for a stock."""
    ticker: str
    timestamp: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    direction_prob: float  # P(price goes up)
    expected_return: Optional[float] = None
    risk_score: float = 0.5  # 0 = low risk, 1 = high risk
    model_agreement: float = 0.0  # fraction of models agreeing
    features_summary: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "direction_prob": round(self.direction_prob, 4),
            "expected_return": round(self.expected_return, 6) if self.expected_return is not None else None,
            "risk_score": round(self.risk_score, 4),
            "model_agreement": round(self.model_agreement, 4),
            "features_summary": self.features_summary,
        }


class SignalGenerator:
    """
    Generates trading signals from ensemble model predictions.

    Signal logic:
    - P(up) > buy_threshold AND effective_confidence > min_confidence → BUY
    - P(up) < sell_threshold AND effective_confidence > min_confidence → SELL
    - Otherwise → HOLD

    Effective confidence = raw probability confidence + agreement bonus.
    This accounts for the fact that a well-calibrated ensemble on a hard
    problem (stock prediction, AUC ~0.52) produces modest probabilities
    (0.45–0.58) even when it has a real edge. Model consensus amplifies
    the signal when 4+ of 6 models agree.

    Thresholds are dynamic based on user's risk tolerance.
    """

    RISK_PROFILES = {
        # Thresholds calibrated for ensemble AUC ~0.52 (proba range 0.45–0.58)
        "low":    {"buy_threshold": 0.56, "sell_threshold": 0.44, "min_confidence": 0.30},
        "medium": {"buy_threshold": 0.53, "sell_threshold": 0.47, "min_confidence": 0.15},
        "high":   {"buy_threshold": 0.51, "sell_threshold": 0.49, "min_confidence": 0.08},
    }

    def __init__(self, risk_tolerance: str = "medium"):
        profile = self.RISK_PROFILES.get(risk_tolerance, self.RISK_PROFILES["medium"])
        self.buy_threshold = profile["buy_threshold"]
        self.sell_threshold = profile["sell_threshold"]
        self.min_confidence = profile["min_confidence"]
        self.risk_tolerance = risk_tolerance

    def generate_signal(
        self,
        ticker: str,
        ensemble_proba: float,
        individual_predictions: Optional[Dict[str, float]] = None,
        expected_return: Optional[float] = None,
        volatility: Optional[float] = None,
        current_features: Optional[Dict] = None,
    ) -> Signal:
        """
        Generate a single signal for a ticker.

        Args:
            ticker: Stock symbol
            ensemble_proba: Calibrated P(up) from ensemble
            individual_predictions: Per-model probabilities
            expected_return: Predicted return magnitude
            volatility: Current/predicted volatility
            current_features: Key features for explainability

        Returns:
            Signal object
        """
        # Raw confidence = distance from 0.5, scaled to [0, 1]
        raw_confidence = abs(ensemble_proba - 0.5) * 2

        # Model agreement
        agreement = 0.0
        if individual_predictions:
            up_votes = sum(1 for p in individual_predictions.values() if p > 0.5)
            agreement = up_votes / len(individual_predictions)

        # Agreement bonus: when most models agree, boost confidence.
        # 6/6 agree → +0.25, 5/6 → +0.15, 4/6 → +0.05, ≤3/6 → 0
        direction_agreement = agreement if ensemble_proba > 0.5 else (1 - agreement)
        if direction_agreement >= 0.9:
            agreement_bonus = 0.25
        elif direction_agreement >= 0.75:
            agreement_bonus = 0.15
        elif direction_agreement >= 0.60:
            agreement_bonus = 0.05
        else:
            agreement_bonus = 0.0

        # Effective confidence combines probability edge + consensus
        confidence = min(1.0, raw_confidence + agreement_bonus)

        # Risk score (based on volatility and disagreement)
        risk_score = 0.5
        if volatility is not None:
            risk_score = min(1.0, volatility / 0.5)  # Normalize annual vol
        if individual_predictions:
            disagreement = 1 - agreement if ensemble_proba > 0.5 else agreement
            risk_score = risk_score * 0.6 + disagreement * 0.4

        # Determine action
        if ensemble_proba > self.buy_threshold and confidence >= self.min_confidence:
            action = "BUY"
        elif ensemble_proba < self.sell_threshold and confidence >= self.min_confidence:
            action = "SELL"
        else:
            action = "HOLD"

        # Feature summary for explainability
        features_summary = {}
        if current_features:
            key_features = [
                "rsi_14", "macd", "sma_cross_50_200", "adx",
                "volatility_20", "drawdown", "regime_trend",
                "volume_ratio_20", "price_zscore_60",
            ]
            for f in key_features:
                if f in current_features:
                    features_summary[f] = round(float(current_features[f]), 4)

        return Signal(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            action=action,
            confidence=confidence,
            direction_prob=ensemble_proba,
            expected_return=expected_return,
            risk_score=risk_score,
            model_agreement=agreement,
            features_summary=features_summary,
        )

    def generate_batch_signals(
        self,
        tickers: List[str],
        ensemble_probas: np.ndarray,
        individual_predictions: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Signal]:
        """Generate signals for multiple tickers."""
        signals = []

        for i, ticker in enumerate(tickers):
            ind_preds = None
            if individual_predictions:
                ind_preds = {k: float(v[i]) for k, v in individual_predictions.items() if i < len(v)}

            signal = self.generate_signal(
                ticker=ticker,
                ensemble_proba=float(ensemble_probas[i]),
                individual_predictions=ind_preds,
            )
            signals.append(signal)

        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)

        return signals
