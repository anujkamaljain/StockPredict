"""
Training pipeline with walk-forward validation and custom loss functions.
Implements time-series-aware cross-validation and Sharpe-optimized training.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from pathlib import Path
import time
from torch.utils.data import Dataset, DataLoader

from app.models.lstm import LSTMModel
from app.models.transformer import TemporalFusionTransformer
from app.models.cnn import MultiScaleCNN
from app.models.tree_models import TreeModels
from app.models.ensemble import EnsembleModel
from app.config import config


# =============================================================================
# Custom Loss Functions
# =============================================================================

class SharpeAwareLoss(nn.Module):
    """
    Custom loss that combines BCE with a Sharpe-ratio penalty.
    Encourages the model to make predictions that would lead to
    higher risk-adjusted returns, not just classification accuracy.
    """

    def __init__(self, alpha: float = 0.5, risk_free_rate: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.rf = risk_free_rate / 252  # Daily

    def forward(
        self,
        direction_prob: torch.Tensor,
        target_direction: torch.Tensor,
        target_return: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Standard BCE loss
        bce_loss = self.bce(direction_prob, target_direction.float())

        if target_return is None or self.alpha == 0:
            return bce_loss

        # Sharpe-like penalty: position * return should have high Sharpe
        position = direction_prob * 2 - 1  # Convert [0,1] to [-1,1]
        strategy_returns = position * target_return

        if strategy_returns.numel() < 2:
            return bce_loss

        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std() + 1e-8
        sharpe = -(mean_ret - self.rf) / std_ret  # Negative because we minimize

        return (1 - self.alpha) * bce_loss + self.alpha * sharpe


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining direction, return, and volatility prediction."""

    def __init__(self, direction_weight: float = 0.5, return_weight: float = 0.3, vol_weight: float = 0.2):
        super().__init__()
        self.direction_weight = direction_weight
        self.return_weight = return_weight
        self.vol_weight = vol_weight
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0.0

        if "direction_prob" in predictions and "direction" in targets:
            loss += self.direction_weight * self.bce(predictions["direction_prob"], targets["direction"].float())

        if "return_pred" in predictions and "return" in targets:
            loss += self.return_weight * self.mse(predictions["return_pred"], targets["return"])

        if "volatility_pred" in predictions and "volatility" in targets:
            valid = ~torch.isnan(targets["volatility"])
            if valid.sum() > 0:
                loss += self.vol_weight * self.mse(
                    predictions["volatility_pred"][valid], targets["volatility"][valid]
                )

        return loss


# =============================================================================
# Dataset
# =============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset for sequence-based models (LSTM, Transformer, CNN)."""

    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[Dict[str, np.ndarray]] = None,
        seq_length: int = 60,
    ):
        self.features = torch.FloatTensor(features)
        self.seq_length = seq_length
        self.targets = {}
        if targets:
            for key, vals in targets.items():
                self.targets[key] = torch.FloatTensor(vals)

    def __len__(self):
        return max(0, len(self.features) - self.seq_length)

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_length]
        target_idx = idx + self.seq_length - 1

        if self.targets:
            y = {key: vals[target_idx] for key, vals in self.targets.items() if target_idx < len(vals)}
            return x, y
        return x


# =============================================================================
# Walk-Forward Validation
# =============================================================================

class WalkForwardValidator:
    """
    Time-series cross-validation using walk-forward (expanding window).
    Prevents look-ahead bias that would plague standard k-fold CV.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_min_pct: float = 0.3,
        gap: int = 5,  # Gap between train and test to prevent leakage
    ):
        self.n_splits = n_splits
        self.train_min_pct = train_min_pct
        self.gap = gap

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits for walk-forward validation.

        Each fold:
        - Train: all data up to split point
        - Gap: skip `gap` samples
        - Test: next chunk of data

        Returns:
            List of (train_indices, test_indices) tuples
        """
        min_train = int(n_samples * self.train_min_pct)
        test_size = (n_samples - min_train - self.gap * self.n_splits) // self.n_splits

        if test_size < 10:
            logger.warning(f"Very small test size: {test_size}. Reducing splits.")
            test_size = max(10, (n_samples - min_train) // 3)
            self.n_splits = max(1, (n_samples - min_train - self.gap) // test_size)

        splits = []
        for i in range(self.n_splits):
            test_end = min_train + (i + 1) * test_size + i * self.gap
            test_start = test_end - test_size
            train_end = test_start - self.gap

            if test_end > n_samples:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))

        logger.info(f"Walk-forward: {len(splits)} splits, test_size={test_size}")
        return splits


# =============================================================================
# Training Pipeline
# =============================================================================

class TrainingPipeline:
    """
    Orchestrates model training for all model types.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or config.model.device)
        self.model_dir = config.data.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training device: {self.device}")

    def train_deep_model(
        self,
        model: nn.Module,
        train_features: np.ndarray,
        train_targets: Dict[str, np.ndarray],
        val_features: np.ndarray,
        val_targets: Dict[str, np.ndarray],
        model_name: str = "model",
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 15,
        seq_length: int = 60,
    ) -> Dict[str, Any]:
        """
        Train a deep learning model with early stopping.

        Returns:
            Training history and best metrics
        """
        model = model.to(self.device)

        # Datasets
        train_dataset = TimeSeriesDataset(train_features, train_targets, seq_length)
        val_dataset = TimeSeriesDataset(val_features, val_targets, seq_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Loss and optimizer
        criterion = MultiTaskLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

        # Training loop
        best_val_loss = float("inf")
        best_epoch = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Train
            model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = {k: v.to(self.device) for k, v in batch_y.items()}

                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()

            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = {k: v.to(self.device) for k, v in batch_y.items()}
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses) if train_losses else 0
            val_loss = np.mean(val_losses) if val_losses else 0
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"[{model_name}] Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), self.model_dir / f"{model_name}_best.pt")
            elif epoch - best_epoch >= patience:
                logger.info(f"[{model_name}] Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        model.load_state_dict(torch.load(self.model_dir / f"{model_name}_best.pt", weights_only=True))
        logger.info(f"[{model_name}] Best epoch: {best_epoch + 1}, val_loss: {best_val_loss:.4f}")

        return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss}

    def predict_deep_model(
        self,
        model: nn.Module,
        features: np.ndarray,
        seq_length: int = 60,
        batch_size: int = 128,
    ) -> Dict[str, np.ndarray]:
        """Get predictions from a trained deep model."""
        model = model.to(self.device)
        model.eval()

        dataset = TimeSeriesDataset(features, seq_length=seq_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions: Dict[str, List] = {}

        with torch.no_grad():
            for batch_x in loader:
                if isinstance(batch_x, tuple):
                    batch_x = batch_x[0]
                batch_x = batch_x.to(self.device)
                preds = model(batch_x)

                for key, val in preds.items():
                    if key not in all_predictions:
                        all_predictions[key] = []
                    all_predictions[key].append(val.cpu().numpy())

        return {key: np.concatenate(vals) for key, vals in all_predictions.items()}

    def train_all_models(
        self,
        X: np.ndarray,
        targets: Dict[str, np.ndarray],
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Train all model types and build ensemble.

        Uses walk-forward validation for time-series aware evaluation.
        """
        n_features = X.shape[1]
        seq_length = config.model.sequence_length
        results = {}

        # Walk-forward split
        wf = WalkForwardValidator(n_splits=3)
        splits = wf.split(len(X))

        if not splits:
            logger.error("Could not create walk-forward splits")
            return results

        # Use last split for final evaluation
        train_idx, test_idx = splits[-1]
        val_size = int(len(train_idx) * 0.15)
        val_idx = train_idx[-val_size:]
        train_idx = train_idx[:-val_size]

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        targets_train = {k: v[train_idx] for k, v in targets.items()}
        targets_val = {k: v[val_idx] for k, v in targets.items()}
        targets_test = {k: v[test_idx] for k, v in targets.items()}

        # ---- 1. LSTM ----
        logger.info("=" * 50 + " Training LSTM " + "=" * 50)
        lstm = LSTMModel(
            input_size=n_features,
            hidden_size=config.model.lstm_hidden_size,
            num_layers=config.model.lstm_num_layers,
            dropout=config.model.lstm_dropout,
        )
        lstm_result = self.train_deep_model(
            lstm, X_train, targets_train, X_val, targets_val,
            model_name="lstm", epochs=config.model.epochs,
            batch_size=config.model.batch_size, lr=config.model.learning_rate,
            seq_length=seq_length,
        )
        results["lstm"] = lstm_result

        # ---- 2. Transformer ----
        logger.info("=" * 50 + " Training Transformer " + "=" * 50)
        transformer = TemporalFusionTransformer(
            input_size=n_features,
            d_model=config.model.transformer_d_model,
            nhead=config.model.transformer_nhead,
            num_encoder_layers=config.model.transformer_num_layers,
            dim_feedforward=config.model.transformer_dim_feedforward,
            dropout=config.model.transformer_dropout,
            seq_length=seq_length,
        )
        tft_result = self.train_deep_model(
            transformer, X_train, targets_train, X_val, targets_val,
            model_name="transformer", epochs=config.model.epochs,
            batch_size=config.model.batch_size, lr=config.model.learning_rate * 0.5,
            seq_length=seq_length,
        )
        results["transformer"] = tft_result

        # ---- 3. CNN ----
        logger.info("=" * 50 + " Training CNN " + "=" * 50)
        cnn = MultiScaleCNN(input_size=n_features, dropout=0.3)
        cnn_result = self.train_deep_model(
            cnn, X_train, targets_train, X_val, targets_val,
            model_name="cnn", epochs=config.model.epochs,
            batch_size=config.model.batch_size, lr=config.model.learning_rate,
            seq_length=seq_length,
        )
        results["cnn"] = cnn_result

        # ---- 4. Tree Models (use flattened features, no sequence) ----
        logger.info("=" * 50 + " Training Tree Models " + "=" * 50)
        y_train = targets_train.get("direction", np.zeros(len(X_train)))
        y_val = targets_val.get("direction", np.zeros(len(X_val)))

        tree = TreeModels(model_dir=self.model_dir)
        try:
            xgb_metrics = tree.train_xgboost(X_train, y_train, X_val, y_val, feature_names)
            results["xgboost"] = xgb_metrics
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")

        try:
            lgb_metrics = tree.train_lightgbm(X_train, y_train, X_val, y_val, feature_names)
            results["lightgbm"] = lgb_metrics
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")

        try:
            cb_metrics = tree.train_catboost(X_train, y_train, X_val, y_val, feature_names)
            results["catboost"] = cb_metrics
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")

        # ---- 5. Ensemble ----
        logger.info("=" * 50 + " Building Ensemble " + "=" * 50)
        ensemble_preds = {}

        # Get predictions from deep models on test set
        for name, model_obj in [("lstm", lstm), ("transformer", transformer), ("cnn", cnn)]:
            try:
                preds = self.predict_deep_model(model_obj, X_test, seq_length)
                if "direction_prob" in preds:
                    ensemble_preds[name] = preds["direction_prob"]
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")

        # Get predictions from tree models
        tree_preds = tree.predict(X_test)
        ensemble_preds.update(tree_preds)

        if len(ensemble_preds) >= 2:
            # Align lengths
            min_len = min(len(v) for v in ensemble_preds.values())
            ensemble_preds = {k: v[:min_len] for k, v in ensemble_preds.items()}
            y_test = targets_test.get("direction", np.zeros(len(X_test)))[:min_len]

            ensemble = EnsembleModel(model_dir=self.model_dir)
            ensemble_metrics = ensemble.fit(ensemble_preds, y_test)
            results["ensemble"] = ensemble_metrics

        logger.info("=" * 50 + " Training Complete " + "=" * 50)
        return results
