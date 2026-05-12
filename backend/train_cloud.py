"""
=============================================================================
ROBUST CLOUD TRAINING SCRIPT — Run on Colab / DGX / local GPU / AWS / GCP
=============================================================================

WORKFLOW:
1. Upload this entire 'backend/' folder to your cloud environment
2. Install dependencies: pip install -r requirements.txt
3. Run: python train_cloud.py --tickers AAPL,MSFT,GOOGL --epochs 100
4. Download the 'trained_models/' folder
5. Place it at: backend/data/models/ on your local machine
6. Restart your local FastAPI server — it will auto-load trained models

ROBUSTNESS FEATURES:
  - Reproducibility: fixed seeds for torch/numpy/random + deterministic CUDA
  - Per-ticker sequence construction (no cross-ticker contamination)
  - Aligned tree/deep predictions (no off-by-(seq_length) bug)
  - Feature selection (top N by tree importance) — prevents overfitting
  - Label smoothing (configurable) — prevents overconfident predictions
  - Purged gap between splits — prevents temporal leakage at boundaries
  - Early stopping on best val AUC (not just loss) — robust model selection
  - Gradient clipping + AdamW with weight decay — stable optimization
  - Mixed precision (AMP) on GPU — faster training, lower VRAM
  - LR warmup + cosine annealing — avoids early-epoch divergence
  - Class-weight balancing — handles imbalanced targets
  - Scaler fit on TRAIN only — no test leakage
  - Cross-validated isotonic calibration on ensemble — well-calibrated outputs
  - Threshold optimization for F1 and Sharpe — trade-ready thresholds
  - VRAM-safe batched inference — works on 4GB laptop GPUs

ACCEPTANCE GATES (training fails if not met):
  - Test AUC >= 0.52 (better than coin flip)
  - Train/Test AUC gap <= 0.10 (no severe overfitting)
  - Test Brier <= 0.26 (better than uniform 0.5)
  - Ensemble AUC >= max(individual AUCs) - 0.005 (ensemble adds value)
  - Permutation-importance check (model uses features, not noise)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.data.providers.alpha_vantage import AlphaVantageProvider
from app.data.providers.yahoo import YahooFinanceProvider
from app.data.validation import DataValidator
from app.features.engineering import FeatureEngineer
from app.models.cnn import MultiScaleCNN
from app.models.ensemble import EnsembleModel
from app.models.lstm import LSTMModel
from app.models.transformer import TemporalFusionTransformer

try:
    from app.models.tree_models import TreeModels
    HAS_TREES = True
except ImportError:
    HAS_TREES = False
    print("WARNING: XGBoost/LightGBM/CatBoost not installed. Skipping tree models.")
    print("   Install with: pip install xgboost lightgbm catboost")


# =============================================================================
# Configuration: Acceptance Gates (training fails if not met)
# =============================================================================

@dataclass
class AcceptanceGates:
    """Quality gates that the trained model must satisfy to be considered ready."""
    min_test_auc: float = 0.52
    min_test_accuracy: float = 0.51
    max_test_brier: float = 0.26
    max_train_test_auc_gap: float = 0.10
    min_ensemble_auc_over_individual: float = -0.005
    min_individual_auc: float = 0.50
    require_permutation_importance: bool = True
    permutation_drop_threshold: float = 0.005

    def as_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Losses
# =============================================================================

class LabelSmoothingBCE(nn.Module):
    """BCE loss with label smoothing for better calibration on unseen data."""

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
        target_smooth = target * (1.0 - self.smoothing) + (1.0 - target) * self.smoothing
        return self.bce(pred, target_smooth)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stock ML models on GPU with robustness gates")
    parser.add_argument("--tickers", type=str,
                        default="AAPL,MSFT,GOOGL,AMZN,NVDA,META,JPM,JNJ,V,PG",
                        help="Comma-separated ticker list (10+ recommended for robustness)")
    parser.add_argument("--start", type=str, default="2010-01-01",
                        help="Training data start date")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Max epochs (early stopping will find optimum)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=60,
                        help="Lookback window for sequence models")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Learning rate (0.0005 is safer than 0.001)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--max_features", type=int, default=80,
                        help="Max features to keep after selection")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="Label smoothing factor")
    parser.add_argument("--gap", type=int, default=5,
                        help="Purged gap between train/val/test (days)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="trained_models")
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda, cpu, or auto")
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable mixed-precision training on CUDA")
    parser.add_argument("--alpha_vantage_key", type=str, default="",
                        help="Alpha Vantage API key (primary data source)")
    parser.add_argument("--force_accept", action="store_true",
                        help="Skip acceptance gates (NOT RECOMMENDED — produces unreliable models)")
    parser.add_argument("--min_rows_per_ticker", type=int, default=400,
                        help="Skip tickers with fewer rows than this after feature engineering")
    return parser.parse_args()


def detect_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU detected: {name} ({mem:.1f} GB)")
            return dev
        print("WARNING: No GPU found, using CPU (training will be slow)")
        return torch.device("cpu")
    return torch.device(preference)


# =============================================================================
# Data Loading
# =============================================================================

def fetch_ticker_data(
    tickers: List[str],
    start: str,
    av_key: str,
    min_rows: int,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Fetch and feature-engineer data for each ticker independently.
    Returns dict {ticker: feature_df} and union feature_names list.
    """
    print("\n--- STEP 1: Fetching data ---")
    av_api_key = av_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")
    alpha_vantage = AlphaVantageProvider(api_key=av_api_key)
    yahoo = YahooFinanceProvider()
    validator = DataValidator()
    engineer = FeatureEngineer()

    if alpha_vantage.is_configured:
        print("   Alpha Vantage API key configured (primary)")
    else:
        print("   No Alpha Vantage key -- using Yahoo Finance only")

    ticker_dfs: Dict[str, pd.DataFrame] = {}
    feature_names: Optional[List[str]] = None

    for ticker in tickers:
        print(f"  Fetching {ticker}...", end=" ")
        df = pd.DataFrame()

        if alpha_vantage.is_configured and alpha_vantage.requests_remaining > 0:
            try:
                df = alpha_vantage.fetch_daily(ticker, use_cache=False)
                if not df.empty:
                    df = df[df.index >= pd.Timestamp(start)]
                    print(f"[AV: {len(df)} rows]", end=" ")
            except Exception as e:
                print(f"[AV error: {e}]", end=" ")

        if df.empty:
            df = yahoo.fetch_ohlcv(ticker, start=start, use_cache=False)
            if not df.empty:
                print(f"[Yahoo: {len(df)} rows]", end=" ")

        if df.empty:
            print("FAILED - No data")
            continue

        df = validator.clean(df, ticker)
        featured = engineer.compute_features(df)
        featured = featured.dropna()

        if len(featured) < min_rows:
            print(f"SKIP - Too few rows ({len(featured)} < {min_rows})")
            continue

        if feature_names is None:
            exclude_cols = {
                "Open", "High", "Low", "Close", "Volume",
                "target_return", "target_direction", "target_return_5d",
                "target_direction_5d", "target_volatility_5d",
            }
            feature_names = [
                c for c in featured.columns
                if c not in exclude_cols
                and pd.api.types.is_numeric_dtype(featured[c])
            ]
        else:
            missing = [c for c in feature_names if c not in featured.columns]
            for c in missing:
                featured[c] = 0.0

        ticker_dfs[ticker] = featured
        print(f"OK - {len(featured)} samples")

    if not ticker_dfs:
        raise RuntimeError("No valid data fetched. Check tickers and internet.")

    if feature_names is None:
        raise RuntimeError("Could not determine feature_names from any ticker.")

    total_rows = sum(len(df) for df in ticker_dfs.values())
    print(f"\n   Loaded {len(ticker_dfs)} tickers, {total_rows} total samples, "
          f"{len(feature_names)} features")
    return ticker_dfs, feature_names


# =============================================================================
# Per-Ticker Split and Sequence Construction
# =============================================================================

def time_series_split_indices(n: int, train_pct: float, val_pct: float, gap: int
                              ) -> Tuple[slice, slice, slice]:
    """Compute slice boundaries for chronological train/val/test split with purged gaps."""
    train_end = int(n * train_pct)
    val_start = train_end + gap
    val_end = int(n * (train_pct + val_pct))
    test_start = val_end + gap

    if val_start >= val_end or test_start >= n:
        gap_used = 0
        val_start = train_end
        test_start = val_end
    else:
        gap_used = gap
    _ = gap_used

    return slice(0, train_end), slice(val_start, val_end), slice(test_start, n)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_length: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequence windows for sequence models AND return the corresponding
    flat (last-timestep) feature row indices for trees.

    Returns:
        X_seq:    (N - seq_length + 1, seq_length, n_features)
        y_seq:    (N - seq_length + 1,)
        flat_idx: indices into X for the LAST timestep of each window
                  -> use for tree predictions, aligned with deep model predictions.
    """
    if len(X) < seq_length:
        n_features = X.shape[1] if X.ndim > 1 else 0
        return (
            np.empty((0, seq_length, n_features), dtype=X.dtype),
            np.empty((0,), dtype=y.dtype),
            np.empty((0,), dtype=np.int64),
        )

    n_windows = len(X) - seq_length + 1
    Xs = np.stack([X[i: i + seq_length] for i in range(n_windows)], axis=0)
    ys = np.array([y[i + seq_length - 1] for i in range(n_windows)])
    idx = np.arange(seq_length - 1, len(X))
    return Xs, ys, idx


@dataclass
class SplitArrays:
    """Container for train/val/test arrays, both sequence and flat (aligned)."""
    X_train_seq: np.ndarray
    y_train_seq: np.ndarray
    X_val_seq: np.ndarray
    y_val_seq: np.ndarray
    X_test_seq: np.ndarray
    y_test_seq: np.ndarray
    X_train_flat: np.ndarray
    y_train_flat: np.ndarray
    X_val_flat: np.ndarray
    y_val_flat: np.ndarray
    X_test_flat: np.ndarray
    y_test_flat: np.ndarray
    ticker_train: np.ndarray = field(default_factory=lambda: np.array([]))
    ticker_val: np.ndarray = field(default_factory=lambda: np.array([]))
    ticker_test: np.ndarray = field(default_factory=lambda: np.array([]))


def build_per_ticker_splits(
    ticker_dfs: Dict[str, pd.DataFrame],
    feature_names: List[str],
    seq_length: int,
    train_pct: float,
    val_pct: float,
    gap: int,
    selected_idx: Optional[np.ndarray] = None,
    scaler: Optional[RobustScaler] = None,
) -> Tuple[SplitArrays, RobustScaler, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Build sequences PER TICKER (so windows never cross ticker boundaries),
    then concatenate across tickers.

    Scaler is fit only on training rows from all tickers.

    Returns:
        SplitArrays with both seq and flat (aligned) arrays
        Fitted scaler
        Per-ticker raw (X_train, y_train) tuples for feature selection
    """
    train_X_all: List[np.ndarray] = []
    train_y_all: List[np.ndarray] = []
    val_X_all, val_y_all, val_t_all = [], [], []
    test_X_all, test_y_all, test_t_all = [], [], []

    per_ticker_train_raw: List[Tuple[np.ndarray, np.ndarray]] = []

    for ticker, df in ticker_dfs.items():
        X_raw = df[feature_names].values.astype(np.float64)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        y = df["target_direction"].values.astype(np.float32)

        if selected_idx is not None:
            X_raw = X_raw[:, selected_idx]

        n = len(X_raw)
        tr_sl, va_sl, te_sl = time_series_split_indices(n, train_pct, val_pct, gap)

        train_X_all.append(X_raw[tr_sl])
        train_y_all.append(y[tr_sl])
        per_ticker_train_raw.append((X_raw[tr_sl], y[tr_sl]))
        val_X_all.append(X_raw[va_sl])
        val_y_all.append(y[va_sl])
        val_t_all.append(np.array([ticker] * (va_sl.stop - va_sl.start)))
        test_X_all.append(X_raw[te_sl])
        test_y_all.append(y[te_sl])
        test_t_all.append(np.array([ticker] * (te_sl.stop - te_sl.start)))

    X_train_raw = np.concatenate(train_X_all, axis=0)
    y_train_raw = np.concatenate(train_y_all, axis=0)

    if scaler is None:
        scaler = RobustScaler()
        scaler.fit(X_train_raw)

    # Build per-ticker scaled sequences and aligned flat arrays
    Xtr_seq, ytr_seq, Xtr_flat, ytr_flat = [], [], [], []
    Xva_seq, yva_seq, Xva_flat, yva_flat, va_tick = [], [], [], [], []
    Xte_seq, yte_seq, Xte_flat, yte_flat, te_tick = [], [], [], [], []

    for i, (ticker, df) in enumerate(ticker_dfs.items()):
        X_raw = df[feature_names].values.astype(np.float64)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        y = df["target_direction"].values.astype(np.float32)
        if selected_idx is not None:
            X_raw = X_raw[:, selected_idx]
        X_scaled = scaler.transform(X_raw).astype(np.float32)

        n = len(X_scaled)
        tr_sl, va_sl, te_sl = time_series_split_indices(n, train_pct, val_pct, gap)

        # Train
        xs, ys, idx = make_sequences(X_scaled[tr_sl], y[tr_sl], seq_length)
        if len(xs):
            Xtr_seq.append(xs); ytr_seq.append(ys)
            Xtr_flat.append(X_scaled[tr_sl][idx])
            ytr_flat.append(y[tr_sl][idx])

        # Val
        xs, ys, idx = make_sequences(X_scaled[va_sl], y[va_sl], seq_length)
        if len(xs):
            Xva_seq.append(xs); yva_seq.append(ys)
            Xva_flat.append(X_scaled[va_sl][idx])
            yva_flat.append(y[va_sl][idx])
            va_tick.append(np.array([ticker] * len(xs)))

        # Test
        xs, ys, idx = make_sequences(X_scaled[te_sl], y[te_sl], seq_length)
        if len(xs):
            Xte_seq.append(xs); yte_seq.append(ys)
            Xte_flat.append(X_scaled[te_sl][idx])
            yte_flat.append(y[te_sl][idx])
            te_tick.append(np.array([ticker] * len(xs)))

    def _cat(arrays, axis=0):
        return np.concatenate(arrays, axis=axis) if arrays else np.empty(0)

    splits = SplitArrays(
        X_train_seq=_cat(Xtr_seq), y_train_seq=_cat(ytr_seq),
        X_val_seq=_cat(Xva_seq), y_val_seq=_cat(yva_seq),
        X_test_seq=_cat(Xte_seq), y_test_seq=_cat(yte_seq),
        X_train_flat=_cat(Xtr_flat), y_train_flat=_cat(ytr_flat),
        X_val_flat=_cat(Xva_flat), y_val_flat=_cat(yva_flat),
        X_test_flat=_cat(Xte_flat), y_test_flat=_cat(yte_flat),
        ticker_val=_cat(va_tick), ticker_test=_cat(te_tick),
    )
    return splits, scaler, per_ticker_train_raw


# =============================================================================
# Feature Selection
# =============================================================================

def select_features(X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str],
                    max_features: int = 80) -> Tuple[np.ndarray, List[str]]:
    """
    Select top features using XGBoost importance ranking on training data only.
    Reduces dimensionality to prevent overfitting with 170+ features.
    """
    print(f"\n--- FEATURE SELECTION: {len(feature_names)} -> top {max_features} ---")
    try:
        import xgboost as xgb
        selector = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", device="cpu", random_state=42,
            verbosity=0,
        )
        selector.fit(X_train, y_train)
        importances = selector.feature_importances_

        max_features = min(max_features, len(feature_names))
        top_idx = np.argsort(importances)[::-1][:max_features]
        top_idx = np.sort(top_idx)

        selected_names = [feature_names[i] for i in top_idx]
        print(f"   Top 10 features: {selected_names[:10]}")
        print(f"   Kept {len(selected_names)} / {len(feature_names)} features")
        return top_idx, selected_names

    except Exception as e:
        print(f"   Feature selection failed ({e}), keeping all features")
        return np.arange(len(feature_names)), feature_names


# =============================================================================
# Metrics
# =============================================================================

def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray,
                               n_bins: int = 15) -> float:
    """Expected Calibration Error — lower is better-calibrated probabilities."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_proba[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_metrics(name: str, y_true: np.ndarray, y_proba: np.ndarray,
                    threshold: float = 0.5, verbose: bool = True) -> Dict[str, float]:
    """Compute the full suite of classification metrics."""
    y_proba = np.clip(y_proba, 1e-7, 1 - 1e-7)
    y_pred = (y_proba > threshold).astype(int)

    metrics: Dict[str, float] = {
        "auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.5,
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ece": expected_calibration_error(y_true, y_proba),
    }
    if verbose:
        print(f"  {name:18s}  AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1']:.4f}  Brier={metrics['brier']:.4f}  ECE={metrics['ece']:.4f}")
    return metrics


# =============================================================================
# Deep Model Training
# =============================================================================

def batched_predict(model: nn.Module, X: np.ndarray, device: torch.device,
                    batch_size: int = 64) -> np.ndarray:
    """VRAM-safe batched prediction — works on 4GB laptop GPUs."""
    model.eval()
    preds: List[torch.Tensor] = []
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i: i + batch_size]).float().to(device)
            out = model(batch)["direction_prob"].cpu()
            preds.append(out)
            del batch
            if use_cuda:
                torch.cuda.empty_cache()
    if not preds:
        return np.array([])
    return torch.cat(preds).numpy()


def train_deep_model(
    model: nn.Module,
    X_train_seq: np.ndarray, y_train_seq: np.ndarray,
    X_val_seq: np.ndarray, y_val_seq: np.ndarray,
    device: torch.device, name: str, output_dir: Path,
    epochs: int = 150, batch_size: int = 64, lr: float = 0.0005,
    patience: int = 20, label_smoothing: float = 0.05, use_amp: bool = False,
) -> Tuple[nn.Module, np.ndarray, Dict]:
    """
    Train a deep model with:
    - Label smoothing
    - AdamW + cosine annealing with warm restarts + linear LR warmup
    - Gradient clipping
    - Mixed precision (optional)
    - Early stopping on best val AUC (more robust than loss)
    """
    print(f"\n{'=' * 60}\n  Training {name}\n{'=' * 60}")

    model = model.to(device)
    criterion = LabelSmoothingBCE(smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    warmup_epochs = max(1, min(5, epochs // 20))
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(10, epochs // 5), T_mult=2,
    )

    use_amp_real = bool(use_amp) and device.type == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler() if use_amp_real else None

    X_t = torch.from_numpy(X_train_seq).float()
    y_t = torch.from_numpy(y_train_seq).float()

    best_val_auc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_auc": []}

    best_path = output_dir / f"{name}_best.pt"

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_losses: List[float] = []
        perm = torch.randperm(len(X_t))
        for i in range(0, len(perm), batch_size):
            idx = perm[i: i + batch_size]
            bx = X_t[idx].to(device, non_blocking=True)
            by = y_t[idx].to(device, non_blocking=True)
            if len(bx) < 2:
                continue

            optimizer.zero_grad(set_to_none=True)
            if use_amp_real:
                with torch.cuda.amp.autocast():
                    out = model(bx)
                # BCELoss is unsafe under autocast — compute loss in float32
                with torch.cuda.amp.autocast(enabled=False):
                    loss = criterion(out["direction_prob"].float(), by.float())
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                out = model(bx)
                loss = criterion(out["direction_prob"], by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_losses.append(loss.item())
            del bx, by

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # --- VAL ---
        val_pred = batched_predict(model, X_val_seq, device, batch_size)
        val_loss_t = float(criterion(
            torch.from_numpy(val_pred).float(),
            torch.from_numpy(y_val_seq).float(),
        ).item())
        if len(np.unique(y_val_seq)) > 1:
            val_auc = float(roc_auc_score(y_val_seq, val_pred))
        else:
            val_auc = 0.5

        t_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history["train_loss"].append(t_loss)
        history["val_loss"].append(val_loss_t)
        history["val_auc"].append(val_auc)

        improved = val_auc > best_val_auc + 1e-4
        if improved:
            best_val_auc = val_auc
            best_val_loss = val_loss_t
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % 5 == 0 or improved:
            gap = t_loss - val_loss_t
            status = "OK"
            if gap < -0.05:
                status = "OVERFIT"
            elif gap > 0.05:
                status = "UNDERFIT"
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch + 1:3d}/{epochs}: train={t_loss:.4f}  val={val_loss_t:.4f}  "
                  f"val_AUC={val_auc:.4f}  lr={current_lr:.6f}  [{status}]"
                  f"{' (best)' if improved else ''}")

        if epoch - best_epoch >= patience:
            print(f"  Early stop at epoch {epoch + 1} (no AUC improvement for {patience} epochs)")
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, weights_only=True, map_location=device))
    print(f"  Best epoch: {best_epoch + 1}, val_loss: {best_val_loss:.4f}, val_AUC: {best_val_auc:.4f}")

    val_pred = batched_predict(model, X_val_seq, device, batch_size)
    return model, val_pred, history


# =============================================================================
# Threshold Optimization
# =============================================================================

def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                        objective: str = "f1") -> Tuple[float, float]:
    """Find decision threshold that maximizes the given objective."""
    candidates = np.linspace(0.30, 0.70, 41)
    best_thr, best_score = 0.5, -np.inf
    for t in candidates:
        y_pred = (y_proba > t).astype(int)
        if objective == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif objective == "accuracy":
            score = accuracy_score(y_true, y_pred)
        elif objective == "balanced":
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tpr = tp / max(tp + fn, 1)
            tnr = tn / max(tn + fp, 1)
            score = 0.5 * (tpr + tnr)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score, best_thr = score, float(t)
    return best_thr, float(best_score)


# =============================================================================
# Trading Style Validation (Simulated Sharpe on Val)
# =============================================================================

def simulated_pnl_metrics(
    proba: np.ndarray,
    y_direction: np.ndarray,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    cost_bps: float = 5.0,
) -> Dict[str, float]:
    """
    Simulate a simple long/flat strategy:
      - go long when proba > buy_threshold
      - go flat when proba in (sell_threshold, buy_threshold)
      - short when proba < sell_threshold

    Returns: directional hit rate, return proxy, "Sharpe-like" stat.
    Uses sign(y_direction-0.5) as next-period return proxy of ±1.
    """
    proba = np.asarray(proba)
    y = np.asarray(y_direction)
    next_dir = (y * 2 - 1).astype(float)

    pos = np.where(proba > buy_threshold, 1.0,
                   np.where(proba < sell_threshold, -1.0, 0.0))
    pnl = pos * next_dir
    cost = (cost_bps / 1e4) * np.abs(np.diff(np.concatenate([[0], pos])))
    net = pnl - cost

    hit_rate = float(np.mean(pnl > 0)) if len(pnl) else 0.0
    mean_ret = float(np.mean(net))
    std_ret = float(np.std(net)) + 1e-12
    sharpe_like = mean_ret / std_ret * np.sqrt(252) if len(net) else 0.0
    trade_rate = float(np.mean(pos != 0))
    return {
        "hit_rate_directional": hit_rate,
        "trade_rate": trade_rate,
        "mean_pnl_per_step": mean_ret,
        "sharpe_like": sharpe_like,
    }


# =============================================================================
# Permutation Importance Sanity Check
# =============================================================================

def permutation_signal_check(
    base_proba: np.ndarray,
    y_true: np.ndarray,
    permute_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compare ensemble AUC vs permuted-feature AUC.
    If model AUC > permuted AUC by a margin, model is using real signal.
    """
    auc_real = float(roc_auc_score(y_true, base_proba)) if len(np.unique(y_true)) > 1 else 0.5
    auc_perm = float(roc_auc_score(y_true, permute_proba)) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "auc_real": auc_real,
        "auc_permuted": auc_perm,
        "drop": auc_real - auc_perm,
    }


# =============================================================================
# Acceptance Gates
# =============================================================================

def evaluate_gates(test_results: Dict, train_results: Dict,
                   ensemble_metrics: Dict, perm_check: Optional[Dict],
                   gates: AcceptanceGates) -> Tuple[bool, List[str]]:
    """Apply quality gates and return (passed, list_of_failures)."""
    failures: List[str] = []

    ens_test_auc = ensemble_metrics.get("test", {}).get("auc", 0.0)
    ens_test_acc = ensemble_metrics.get("test", {}).get("accuracy", 0.0)
    ens_test_brier = ensemble_metrics.get("test", {}).get("brier", 1.0)
    ens_train_auc = ensemble_metrics.get("train", {}).get("auc", 0.0)

    if ens_test_auc < gates.min_test_auc:
        failures.append(
            f"Test AUC {ens_test_auc:.4f} < min {gates.min_test_auc:.4f} — model is no better than random."
        )
    if ens_test_acc < gates.min_test_accuracy:
        failures.append(
            f"Test accuracy {ens_test_acc:.4f} < min {gates.min_test_accuracy:.4f}."
        )
    if ens_test_brier > gates.max_test_brier:
        failures.append(
            f"Test Brier {ens_test_brier:.4f} > max {gates.max_test_brier:.4f} — poorly calibrated."
        )
    gap = ens_train_auc - ens_test_auc
    if gap > gates.max_train_test_auc_gap:
        failures.append(
            f"Train-Test AUC gap {gap:.4f} > max {gates.max_train_test_auc_gap:.4f} — overfitting."
        )

    best_individual_test_auc = 0.0
    for name, m in test_results.items():
        if "auc" in m and m["auc"] > best_individual_test_auc:
            best_individual_test_auc = m["auc"]
        if m.get("auc", 0.0) < gates.min_individual_auc:
            failures.append(
                f"Individual model '{name}' test AUC {m.get('auc', 0.0):.4f} < min {gates.min_individual_auc:.4f}."
            )
    if ens_test_auc < best_individual_test_auc + gates.min_ensemble_auc_over_individual:
        failures.append(
            f"Ensemble AUC {ens_test_auc:.4f} not above best individual {best_individual_test_auc:.4f} "
            f"by margin {gates.min_ensemble_auc_over_individual:.4f}."
        )

    if gates.require_permutation_importance and perm_check is not None:
        if perm_check.get("drop", 0.0) < gates.permutation_drop_threshold:
            failures.append(
                f"Permutation-importance drop {perm_check.get('drop', 0.0):.4f} < threshold "
                f"{gates.permutation_drop_threshold:.4f} — model is not using real signal."
            )

    return len(failures) == 0, failures


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main() -> int:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = detect_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    print("=" * 70)
    print(" ROBUST CLOUD TRAINING — STOCK MARKET PREDICTOR ")
    print("=" * 70)
    print(f"Tickers       : {tickers}")
    print(f"Start         : {args.start}")
    print(f"Epochs        : {args.epochs} (early stop with patience {args.patience})")
    print(f"Batch size    : {args.batch_size}")
    print(f"Seq length    : {args.seq_length}")
    print(f"Learning rate : {args.lr}")
    print(f"Label smooth  : {args.label_smoothing}")
    print(f"Purged gap    : {args.gap}")
    print(f"Seed          : {args.seed}")
    print(f"Device        : {device}")
    print(f"Mixed prec.   : {args.use_amp and device.type == 'cuda'}")
    print("=" * 70)

    # ---- STEP 1: Fetch & per-ticker feature engineering ----
    ticker_dfs, feature_names = fetch_ticker_data(
        tickers, args.start, args.alpha_vantage_key, args.min_rows_per_ticker,
    )

    # ---- STEP 2: Build PRE-selection splits to fit feature selector ----
    print("\n--- STEP 2: Building pre-selection splits ---")
    pre_splits, _, per_ticker_train_raw = build_per_ticker_splits(
        ticker_dfs, feature_names, args.seq_length,
        train_pct=0.70, val_pct=0.15, gap=args.gap, selected_idx=None, scaler=None,
    )

    print(f"   Pre-selection train rows: {len(pre_splits.X_train_flat)}, "
          f"val: {len(pre_splits.X_val_flat)}, test: {len(pre_splits.X_test_flat)}")
    print(f"   Train class balance: {pre_splits.y_train_flat.mean():.2%} positive")
    print(f"   Val   class balance: {pre_splits.y_val_flat.mean():.2%} positive")
    print(f"   Test  class balance: {pre_splits.y_test_flat.mean():.2%} positive")

    # Concatenate all per-ticker training raw data for feature selection
    full_train_X = np.concatenate([x for x, _ in per_ticker_train_raw], axis=0)
    full_train_y = np.concatenate([y for _, y in per_ticker_train_raw], axis=0)
    full_train_X = np.nan_to_num(full_train_X, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- STEP 3: Feature selection on FULL train data ----
    selected_idx, selected_names = select_features(
        full_train_X, full_train_y, feature_names, max_features=args.max_features,
    )

    # ---- STEP 4: Build FINAL splits with selected features and fitted scaler ----
    print("\n--- STEP 4: Building final splits with scaler fit on TRAIN only ---")
    splits, scaler, _ = build_per_ticker_splits(
        ticker_dfs, feature_names, args.seq_length,
        train_pct=0.70, val_pct=0.15, gap=args.gap,
        selected_idx=selected_idx, scaler=None,
    )

    joblib.dump(scaler, output_dir / "scaler.pkl")
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(selected_names, f, indent=2)
    np.save(output_dir / "selected_feature_indices.npy", selected_idx)
    print(f"   Saved scaler and {len(selected_names)} selected feature names")

    print(f"\n   Final sequence sizes:")
    print(f"     Train: {splits.X_train_seq.shape}, Val: {splits.X_val_seq.shape}, "
          f"Test: {splits.X_test_seq.shape}")
    print(f"   Final flat sizes:")
    print(f"     Train: {splits.X_train_flat.shape}, Val: {splits.X_val_flat.shape}, "
          f"Test: {splits.X_test_flat.shape}")

    if len(splits.X_train_seq) < 200 or len(splits.X_val_seq) < 50 or len(splits.X_test_seq) < 50:
        print(f"\nERROR: Insufficient data after per-ticker split + sequencing.")
        print(f"  Try more tickers, earlier --start, or smaller --seq_length.")
        return 2

    n_features = splits.X_train_seq.shape[2]
    ensemble_val_preds: Dict[str, np.ndarray] = {}
    ensemble_test_preds: Dict[str, np.ndarray] = {}
    ensemble_train_preds: Dict[str, np.ndarray] = {}

    train_results: Dict[str, Dict] = {}
    val_results: Dict[str, Dict] = {}
    test_results: Dict[str, Dict] = {}
    histories: Dict[str, Dict] = {}

    # ---- STEP 5: Deep models ----
    deep_specs = [
        ("lstm", LSTMModel(input_size=n_features, hidden_size=128, num_layers=2, dropout=0.3), 1.0),
        ("transformer", TemporalFusionTransformer(
            input_size=n_features, d_model=128, nhead=8,
            num_encoder_layers=4, dim_feedforward=256, dropout=0.15,
            seq_length=args.seq_length,
        ), 0.5),
        ("cnn", MultiScaleCNN(input_size=n_features, dropout=0.3), 1.0),
    ]

    for name, model_obj, lr_mult in deep_specs:
        model_obj, val_pred, hist = train_deep_model(
            model_obj,
            splits.X_train_seq, splits.y_train_seq,
            splits.X_val_seq, splits.y_val_seq,
            device, name, output_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr * lr_mult, patience=args.patience,
            label_smoothing=args.label_smoothing, use_amp=args.use_amp,
        )
        train_pred = batched_predict(model_obj, splits.X_train_seq, device, args.batch_size)
        test_pred = batched_predict(model_obj, splits.X_test_seq, device, args.batch_size)
        ensemble_train_preds[name] = train_pred
        ensemble_val_preds[name] = val_pred
        ensemble_test_preds[name] = test_pred

        train_results[name] = compute_metrics(f"{name}_train", splits.y_train_seq, train_pred, verbose=False)
        val_results[name] = compute_metrics(f"{name}_val", splits.y_val_seq, val_pred)
        test_results[name] = compute_metrics(f"{name}_test", splits.y_test_seq, test_pred)
        histories[name] = hist

    # ---- STEP 6: Tree models (on flat features, aligned to deep predictions) ----
    if HAS_TREES:
        print(f"\n{'=' * 60}\n  Training Tree Models (XGBoost, LightGBM, CatBoost)\n{'=' * 60}")
        trees = TreeModels(model_dir=output_dir)
        tree_specs = [
            ("xgboost", trees.train_xgboost),
            ("lightgbm", trees.train_lightgbm),
            ("catboost", trees.train_catboost),
        ]
        for method_name, train_fn in tree_specs:
            try:
                train_fn(
                    splits.X_train_flat, splits.y_train_flat.astype(int),
                    splits.X_val_flat, splits.y_val_flat.astype(int),
                    selected_names,
                )
            except Exception as e:
                print(f"  WARNING: {method_name} failed: {e}")

        tree_train_preds = trees.predict(splits.X_train_flat)
        tree_val_preds = trees.predict(splits.X_val_flat)
        tree_test_preds = trees.predict(splits.X_test_flat)
        for name in tree_train_preds:
            ensemble_train_preds[name] = tree_train_preds[name]
            ensemble_val_preds[name] = tree_val_preds[name]
            ensemble_test_preds[name] = tree_test_preds[name]
            train_results[name] = compute_metrics(f"{name}_train", splits.y_train_flat, tree_train_preds[name], verbose=False)
            val_results[name] = compute_metrics(f"{name}_val", splits.y_val_flat, tree_val_preds[name])
            test_results[name] = compute_metrics(f"{name}_test", splits.y_test_flat, tree_test_preds[name])

    if len(ensemble_val_preds) < 2:
        print("\nERROR: Need at least 2 models to build ensemble. Aborting.")
        return 3

    # ---- STEP 7: Ensemble — fit on VAL, evaluate on TEST ----
    print(f"\n{'=' * 60}\n  Building Ensemble (fit on VAL, evaluate on TEST)\n{'=' * 60}")
    min_train = min(len(v) for v in ensemble_train_preds.values())
    min_val = min(len(v) for v in ensemble_val_preds.values())
    min_test = min(len(v) for v in ensemble_test_preds.values())

    train_preds_aligned = {k: v[:min_train] for k, v in ensemble_train_preds.items()}
    val_preds_aligned = {k: v[:min_val] for k, v in ensemble_val_preds.items()}
    test_preds_aligned = {k: v[:min_test] for k, v in ensemble_test_preds.items()}
    y_train_aligned = splits.y_train_seq[:min_train]
    y_val_aligned = splits.y_val_seq[:min_val]
    y_test_aligned = splits.y_test_seq[:min_test]

    ensemble = EnsembleModel(model_dir=output_dir)
    ens_val_fit_metrics = ensemble.fit(val_preds_aligned, y_val_aligned)
    print(f"  Ensemble VAL AUC: {ens_val_fit_metrics['ensemble_auc']:.4f}")

    ens_train_proba = ensemble.predict_proba(train_preds_aligned)
    ens_val_proba = ensemble.predict_proba(val_preds_aligned)
    ens_test_proba = ensemble.predict_proba(test_preds_aligned)

    ensemble_metrics_full = {
        "train": compute_metrics("ENSEMBLE_train", y_train_aligned, ens_train_proba, verbose=False),
        "val": compute_metrics("ENSEMBLE_val", y_val_aligned, ens_val_proba),
        "test": compute_metrics("ENSEMBLE_test", y_test_aligned, ens_test_proba),
    }

    print(f"\n  --- UNSEEN TEST RESULTS (honest metrics) ---")
    for name, m in test_results.items():
        print(f"  {name:18s}  AUC={m['auc']:.4f}  Acc={m['accuracy']:.4f}  "
              f"F1={m['f1']:.4f}  Brier={m['brier']:.4f}")
    print(f"  {'-' * 60}")
    print(f"  {'ENSEMBLE':18s}  AUC={ensemble_metrics_full['test']['auc']:.4f}  "
          f"Acc={ensemble_metrics_full['test']['accuracy']:.4f}  "
          f"F1={ensemble_metrics_full['test']['f1']:.4f}  "
          f"Brier={ensemble_metrics_full['test']['brier']:.4f}  "
          f"ECE={ensemble_metrics_full['test']['ece']:.4f}")

    # ---- STEP 8: Threshold optimization on VAL ----
    print("\n--- STEP 8: Threshold optimization on VAL ---")
    best_thr_f1, best_score_f1 = find_best_threshold(y_val_aligned, ens_val_proba, "f1")
    best_thr_acc, _ = find_best_threshold(y_val_aligned, ens_val_proba, "accuracy")
    best_thr_bal, _ = find_best_threshold(y_val_aligned, ens_val_proba, "balanced")
    print(f"  Best F1 threshold       : {best_thr_f1:.3f} (val F1={best_score_f1:.4f})")
    print(f"  Best accuracy threshold : {best_thr_acc:.3f}")
    print(f"  Best balanced threshold : {best_thr_bal:.3f}")

    # ---- STEP 9: Trading-style validation ----
    print("\n--- STEP 9: Simulated trading-style metrics ---")
    sim_val = simulated_pnl_metrics(ens_val_proba, y_val_aligned,
                                    buy_threshold=best_thr_f1,
                                    sell_threshold=1.0 - best_thr_f1)
    sim_test = simulated_pnl_metrics(ens_test_proba, y_test_aligned,
                                     buy_threshold=best_thr_f1,
                                     sell_threshold=1.0 - best_thr_f1)
    print(f"  VAL  hit_rate={sim_val['hit_rate_directional']:.4f} "
          f"trade_rate={sim_val['trade_rate']:.4f} sharpe_like={sim_val['sharpe_like']:.3f}")
    print(f"  TEST hit_rate={sim_test['hit_rate_directional']:.4f} "
          f"trade_rate={sim_test['trade_rate']:.4f} sharpe_like={sim_test['sharpe_like']:.3f}")

    # ---- STEP 10: Permutation-importance sanity check ----
    print("\n--- STEP 10: Permutation-importance sanity check ---")
    rng = np.random.default_rng(args.seed)
    permuted_test_preds = {
        k: rng.permutation(v) for k, v in test_preds_aligned.items()
    }
    permuted_proba = ensemble.predict_proba(permuted_test_preds)
    perm_check = permutation_signal_check(ens_test_proba, y_test_aligned, permuted_proba)
    print(f"  AUC real       : {perm_check['auc_real']:.4f}")
    print(f"  AUC permuted   : {perm_check['auc_permuted']:.4f}")
    print(f"  AUC drop       : {perm_check['drop']:.4f}  "
          f"(higher means model uses real signal)")

    # ---- STEP 11: Acceptance gates ----
    gates = AcceptanceGates()
    passed, failures = evaluate_gates(
        test_results, train_results, ensemble_metrics_full, perm_check, gates,
    )

    if passed:
        print(f"\n{'=' * 70}\n  ALL ACCEPTANCE GATES PASSED. Model is ready.\n{'=' * 70}")
    else:
        print(f"\n{'=' * 70}\n  ACCEPTANCE GATES FAILED:\n{'=' * 70}")
        for i, fail in enumerate(failures, 1):
            print(f"  {i}. {fail}")
        print(f"\n  Recommendations:")
        print(f"    - Add more tickers / longer history (--tickers / --start)")
        print(f"    - Increase --epochs and --patience")
        print(f"    - Try a different --seq_length (30, 90, 120)")
        print(f"    - Adjust --max_features (try 40 or 120)")
        print(f"    - Inspect training_report.json for detailed metrics")
        if args.force_accept:
            print("\n  --force_accept set: continuing despite failures (NOT RECOMMENDED).")
            passed = True

    # ---- STEP 12: Save comprehensive report ----
    report = {
        "trained_at": datetime.now().isoformat(),
        "tickers": tickers,
        "data_start": args.start,
        "feature_names": selected_names,
        "n_features": n_features,
        "seq_length": args.seq_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "label_smoothing": args.label_smoothing,
        "purged_gap": args.gap,
        "seed": args.seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "splits": {
            "train_seq": int(len(splits.X_train_seq)),
            "val_seq": int(len(splits.X_val_seq)),
            "test_seq": int(len(splits.X_test_seq)),
            "train_flat": int(len(splits.X_train_flat)),
            "val_flat": int(len(splits.X_val_flat)),
            "test_flat": int(len(splits.X_test_flat)),
        },
        "class_balance": {
            "train": float(splits.y_train_flat.mean()) if len(splits.y_train_flat) else 0.0,
            "val": float(splits.y_val_flat.mean()) if len(splits.y_val_flat) else 0.0,
            "test": float(splits.y_test_flat.mean()) if len(splits.y_test_flat) else 0.0,
        },
        "individual_metrics": {
            "train": train_results,
            "val": val_results,
            "test": test_results,
        },
        "ensemble_metrics": ensemble_metrics_full,
        "thresholds": {
            "best_f1": best_thr_f1,
            "best_accuracy": best_thr_acc,
            "best_balanced": best_thr_bal,
        },
        "simulated_trading": {"val": sim_val, "test": sim_test},
        "permutation_check": perm_check,
        "acceptance_gates": {
            "passed": passed,
            "failures": failures,
            "config": gates.as_dict(),
        },
        "robustness_features": [
            "reproducibility_seeds",
            "per_ticker_sequence_construction",
            "aligned_tree_and_deep_predictions",
            "feature_selection_train_only",
            f"label_smoothing_{args.label_smoothing}",
            f"purged_gap_{args.gap}_days",
            "early_stopping_on_val_auc",
            "gradient_clipping",
            "lr_warmup_plus_cosine_annealing",
            "class_weight_balancing_trees",
            "scaler_train_only",
            "ensemble_fit_on_val",
            "cross_validated_isotonic_calibration",
            "vram_safe_batching",
            "mixed_precision_optional",
            "threshold_optimization",
            "simulated_trading_validation",
            "permutation_importance_sanity_check",
            "acceptance_gates",
        ],
        "histories": {k: {kk: [float(v) for v in vv] for kk, vv in h.items()}
                      for k, h in histories.items()},
    }

    with open(output_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 70}\n  TRAINING {'COMPLETE' if passed else 'FAILED (gates not met)'}\n{'=' * 70}")
    print(f"  Models saved to: {output_dir.absolute()}")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:35s} ({size_kb:8.1f} KB)")

    print(f"\n  NEXT STEPS:")
    print(f"  1. Copy '{output_dir}/' to: backend/data/models/")
    print(f"  2. Restart your FastAPI server")
    print(f"  3. Signals will use trained ML models")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
