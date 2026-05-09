"""
Meta-Ensemble Learner.
Combines predictions from LSTM, Transformer, CNN, and tree models
using a stacking approach with a logistic regression meta-learner.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib
from pathlib import Path


class EnsembleModel:
    """
    Stacking meta-ensemble that combines diverse model predictions.

    Strategy:
    1. Each base model produces P(up) probability
    2. Meta-learner (calibrated logistic regression) learns optimal weighting
    3. Output is calibrated probability with confidence score

    Calibration is critical — raw model probabilities are often poorly calibrated
    for financial data due to non-stationarity.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir
        self.meta_learner = None
        self.calibrator = None
        self.base_model_names: List[str] = []
        self.weights: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        base_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        calibrate: bool = True,
    ) -> Dict[str, float]:
        """
        Fit the meta-learner on base model predictions.

        Args:
            base_predictions: dict mapping model_name -> array of P(up)
            y_true: true binary labels
            calibrate: whether to calibrate the final output

        Returns:
            dict with ensemble metrics
        """
        self.base_model_names = sorted(base_predictions.keys())

        # Stack predictions into matrix
        X_meta = np.column_stack(
            [base_predictions[name] for name in self.base_model_names]
        )

        # Add interaction features
        X_augmented = self._augment_features(X_meta)

        # Train meta-learner
        self.meta_learner = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        self.meta_learner.fit(X_augmented, y_true)

        # Extract learned weights
        self.weights = self.meta_learner.coef_[0][: len(self.base_model_names)]
        weight_dict = dict(zip(self.base_model_names, self.weights))
        logger.info(f"Ensemble weights: {weight_dict}")

        # Calibrate
        if calibrate:
            self.calibrator = CalibratedClassifierCV(
                self.meta_learner, method="isotonic", cv=3
            )
            self.calibrator.fit(X_augmented, y_true)

        self._fitted = True

        # Metrics
        ensemble_pred = self.predict_proba(base_predictions)
        metrics = {
            "ensemble_auc": float(roc_auc_score(y_true, ensemble_pred)),
            "ensemble_brier": float(brier_score_loss(y_true, ensemble_pred)),
            "ensemble_accuracy": float(((ensemble_pred > 0.5) == y_true).mean()),
        }

        # Compare with individual models
        for name, pred in base_predictions.items():
            metrics[f"{name}_auc"] = float(roc_auc_score(y_true, pred))
            metrics[f"{name}_brier"] = float(brier_score_loss(y_true, pred))

        logger.info(f"Ensemble AUC: {metrics['ensemble_auc']:.4f}")

        # Save
        if self.model_dir:
            self.save(self.model_dir)

        return metrics

    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get calibrated ensemble probability.

        Args:
            base_predictions: dict mapping model_name -> P(up) array

        Returns:
            Calibrated P(up) array
        """
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        X_meta = np.column_stack(
            [base_predictions.get(name, np.zeros(len(list(base_predictions.values())[0])))
             for name in self.base_model_names]
        )
        X_augmented = self._augment_features(X_meta)

        if self.calibrator is not None:
            return self.calibrator.predict_proba(X_augmented)[:, 1]
        else:
            return self.meta_learner.predict_proba(X_augmented)[:, 1]

    def get_confidence(self, ensemble_proba: np.ndarray) -> np.ndarray:
        """
        Compute confidence score from ensemble probability.

        Confidence = distance from decision boundary (0.5).
        Ranges from 0 (no confidence) to 1 (maximum confidence).
        """
        return np.abs(ensemble_proba - 0.5) * 2

    def get_signal(
        self,
        ensemble_proba: np.ndarray,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
    ) -> np.ndarray:
        """
        Convert probability to action signal.

        Args:
            ensemble_proba: calibrated P(up)
            buy_threshold: minimum probability for buy signal
            sell_threshold: maximum probability for sell signal

        Returns:
            Array of signals: 1 (buy), -1 (sell), 0 (hold)
        """
        signals = np.zeros_like(ensemble_proba, dtype=int)
        signals[ensemble_proba > buy_threshold] = 1
        signals[ensemble_proba < sell_threshold] = -1
        return signals

    def _augment_features(self, X: np.ndarray) -> np.ndarray:
        """Add interaction and statistical features to meta-learner input."""
        features = [X]

        # Mean and std across models
        features.append(X.mean(axis=1, keepdims=True))
        features.append(X.std(axis=1, keepdims=True))

        # Min and max
        features.append(X.min(axis=1, keepdims=True))
        features.append(X.max(axis=1, keepdims=True))

        # Agreement: how many models agree on direction
        agreement = ((X > 0.5).sum(axis=1, keepdims=True) / X.shape[1])
        features.append(agreement)

        return np.hstack(features)

    def save(self, path: Path):
        """Save ensemble model."""
        path.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "meta_learner": self.meta_learner,
            "calibrator": self.calibrator,
            "base_model_names": self.base_model_names,
            "weights": self.weights,
        }
        joblib.dump(save_dict, path / "ensemble_model.pkl")
        logger.info(f"Ensemble saved to {path}")

    def load(self, path: Path):
        """Load ensemble model."""
        save_dict = joblib.load(path / "ensemble_model.pkl")
        self.meta_learner = save_dict["meta_learner"]
        self.calibrator = save_dict["calibrator"]
        self.base_model_names = save_dict["base_model_names"]
        self.weights = save_dict["weights"]
        self._fitted = True
        logger.info(f"Ensemble loaded from {path}")
