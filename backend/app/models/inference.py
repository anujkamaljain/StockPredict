"""
Production inference engine for trained ensemble.

Loads all trained models from `data/models/` (or the configured `model_dir`)
once on first use, caches them in memory, and produces well-calibrated
ensemble probabilities for live signal generation.

Falls back gracefully (returns None) if any required artifact is missing —
the signal route then uses the statistical fallback.

Thread-safe singleton: a single inferencer is shared across FastAPI workers.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger

from app.config import config
from app.models.cnn import MultiScaleCNN
from app.models.ensemble import EnsembleModel
from app.models.lstm import LSTMModel
from app.models.transformer import TemporalFusionTransformer


# Architecture defaults — must match train_cloud.py.
# If train_cloud.py changes hyperparameters, mirror them here.
_LSTM_HIDDEN = 128
_LSTM_LAYERS = 2
_LSTM_DROPOUT = 0.3

_TFT_D_MODEL = 128
_TFT_NHEAD = 8
_TFT_LAYERS = 4
_TFT_FF = 256
_TFT_DROPOUT = 0.15

_CNN_DROPOUT = 0.3

_DEFAULT_SEQ_LENGTH = 60


class ModelInferencer:
    """
    Lazy-loaded inference engine.

    On first call to is_available() / predict_latest(), tries to load:
      - scaler.pkl                       (required)
      - feature_names.json               (required)
      - ensemble_model.pkl               (required)
      - training_report.json             (optional — for seq_length and thresholds)
      - lstm_best.pt                     (optional but recommended)
      - transformer_best.pt              (optional but recommended)
      - cnn_best.pt                      (optional but recommended)
      - xgboost_model.json               (optional)
      - lightgbm_model.pkl               (optional)
      - catboost_model.cbm               (optional)

    Predictions need at least 2 base models present. Otherwise inference
    returns None and the caller falls back to the statistical heuristic.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else config.data.model_dir
        self._lock = threading.Lock()
        self._loaded = False
        self._available = False

        self.scaler = None
        self.feature_names: List[str] = []
        self.seq_length: int = _DEFAULT_SEQ_LENGTH

        self.lstm: Optional[LSTMModel] = None
        self.tft: Optional[TemporalFusionTransformer] = None
        self.cnn: Optional[MultiScaleCNN] = None
        self.xgb = None
        self.lgb = None
        self.cat = None
        self.ensemble: Optional[EnsembleModel] = None

        self.thresholds: Dict[str, float] = {"buy": 0.55, "sell": 0.45}
        self.report_summary: Dict = {}
        self.device = torch.device("cpu")  # Inference always on CPU for low-latency API

    def is_available(self) -> bool:
        """Return True iff trained models can be loaded and used."""
        if not self._loaded:
            self._try_load()
        return self._available

    def info(self) -> Dict:
        """Return a small dict describing what is loaded (for debug endpoints)."""
        self.is_available()
        return {
            "available": self._available,
            "model_dir": str(self.model_dir),
            "n_features": len(self.feature_names),
            "seq_length": self.seq_length,
            "loaded_models": [
                name for name, obj in [
                    ("lstm", self.lstm), ("transformer", self.tft), ("cnn", self.cnn),
                    ("xgboost", self.xgb), ("lightgbm", self.lgb), ("catboost", self.cat),
                ] if obj is not None
            ],
            "thresholds": self.thresholds,
            "report": self.report_summary,
        }

    def _try_load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True  # don't retry on every request

            try:
                scaler_p = self.model_dir / "scaler.pkl"
                features_p = self.model_dir / "feature_names.json"
                ens_p = self.model_dir / "ensemble_model.pkl"

                if not (scaler_p.exists() and features_p.exists() and ens_p.exists()):
                    logger.info(
                        f"ModelInferencer: trained models not found in {self.model_dir}. "
                        f"Falling back to statistical signal."
                    )
                    return

                self.scaler = joblib.load(scaler_p)
                with open(features_p) as f:
                    self.feature_names = json.load(f)
                n_features = len(self.feature_names)
                if n_features == 0:
                    logger.warning("feature_names.json is empty; inference disabled.")
                    return

                report_p = self.model_dir / "training_report.json"
                if report_p.exists():
                    try:
                        with open(report_p) as f:
                            report = json.load(f)
                        self.seq_length = int(report.get("seq_length", _DEFAULT_SEQ_LENGTH))
                        th = report.get("thresholds", {}) or {}
                        buy_t = th.get("best_balanced") or th.get("best_f1") or 0.55
                        buy_t = float(np.clip(buy_t, 0.30, 0.70))
                        self.thresholds = {"buy": buy_t, "sell": float(1.0 - buy_t)}
                        self.report_summary = {
                            "trained_at": report.get("trained_at"),
                            "ensemble_test_auc": report.get("ensemble_metrics", {})
                                                       .get("test", {}).get("auc"),
                            "ensemble_test_brier": report.get("ensemble_metrics", {})
                                                         .get("test", {}).get("brier"),
                            "gates_passed": report.get("acceptance_gates", {}).get("passed"),
                            "tickers": report.get("tickers"),
                        }
                    except Exception as e:
                        logger.warning(f"Failed to parse training_report.json: {e}")

                # Deep models
                lstm_p = self.model_dir / "lstm_best.pt"
                if lstm_p.exists():
                    try:
                        m = LSTMModel(input_size=n_features, hidden_size=_LSTM_HIDDEN,
                                      num_layers=_LSTM_LAYERS, dropout=_LSTM_DROPOUT)
                        m.load_state_dict(torch.load(lstm_p, map_location=self.device,
                                                     weights_only=True))
                        m.eval()
                        self.lstm = m
                    except Exception as e:
                        logger.warning(f"Failed to load LSTM: {e}")

                tft_p = self.model_dir / "transformer_best.pt"
                if tft_p.exists():
                    try:
                        m = TemporalFusionTransformer(
                            input_size=n_features, d_model=_TFT_D_MODEL, nhead=_TFT_NHEAD,
                            num_encoder_layers=_TFT_LAYERS, dim_feedforward=_TFT_FF,
                            dropout=_TFT_DROPOUT, seq_length=self.seq_length,
                        )
                        m.load_state_dict(torch.load(tft_p, map_location=self.device,
                                                     weights_only=True))
                        m.eval()
                        self.tft = m
                    except Exception as e:
                        logger.warning(f"Failed to load Transformer: {e}")

                cnn_p = self.model_dir / "cnn_best.pt"
                if cnn_p.exists():
                    try:
                        m = MultiScaleCNN(input_size=n_features, dropout=_CNN_DROPOUT)
                        m.load_state_dict(torch.load(cnn_p, map_location=self.device,
                                                     weights_only=True))
                        m.eval()
                        self.cnn = m
                    except Exception as e:
                        logger.warning(f"Failed to load CNN: {e}")

                # Tree models
                xgb_p = self.model_dir / "xgboost_model.json"
                if xgb_p.exists():
                    try:
                        import xgboost as xgb_mod
                        clf = xgb_mod.XGBClassifier()
                        clf.load_model(str(xgb_p))
                        self.xgb = clf
                    except Exception as e:
                        logger.warning(f"Failed to load XGBoost: {e}")

                lgb_p = self.model_dir / "lightgbm_model.pkl"
                if lgb_p.exists():
                    try:
                        self.lgb = joblib.load(lgb_p)
                    except Exception as e:
                        logger.warning(f"Failed to load LightGBM: {e}")

                cb_p = self.model_dir / "catboost_model.cbm"
                if cb_p.exists():
                    try:
                        from catboost import CatBoostClassifier
                        cb = CatBoostClassifier()
                        cb.load_model(str(cb_p))
                        self.cat = cb
                    except Exception as e:
                        logger.warning(f"Failed to load CatBoost: {e}")

                # Ensemble (required)
                try:
                    self.ensemble = EnsembleModel(model_dir=self.model_dir)
                    self.ensemble.load(self.model_dir)
                except Exception as e:
                    logger.error(f"Failed to load ensemble: {e}")
                    return

                loaded = [n for n, o in [
                    ("LSTM", self.lstm), ("TFT", self.tft), ("CNN", self.cnn),
                    ("XGB", self.xgb), ("LGB", self.lgb), ("CAT", self.cat),
                ] if o is not None]
                if len(loaded) < 2:
                    logger.warning(
                        f"Only {len(loaded)} base model(s) loaded ({loaded}); "
                        "need >=2 for ensemble. Disabling inference."
                    )
                    return

                self._available = True
                logger.info(
                    f"ModelInferencer ready: {n_features} features, seq_length={self.seq_length}, "
                    f"models={loaded}, thresholds={self.thresholds}, "
                    f"gates_passed={self.report_summary.get('gates_passed')}"
                )

            except Exception as e:
                logger.error(f"ModelInferencer load error: {e}", exc_info=True)
                self._available = False

    def _build_feature_matrix(self, featured_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Project featured_df onto trained feature_names, scale, and return matrix."""
        cols, missing = [], []
        n_rows = len(featured_df)
        for name in self.feature_names:
            if name in featured_df.columns:
                cols.append(np.asarray(featured_df[name].values, dtype=np.float64))
            else:
                missing.append(name)
                cols.append(np.zeros(n_rows, dtype=np.float64))
        if missing:
            logger.warning(
                f"Missing {len(missing)} features at inference (first 5: {missing[:5]}). "
                "Filled with zeros — retrain or align features."
            )
        X = np.column_stack(cols)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X).astype(np.float32)

    def predict_latest(self, featured_df: pd.DataFrame) -> Optional[Dict]:
        """
        Predict for the latest row of featured_df.

        Returns:
            None if inference is unavailable or there's not enough history.
            Otherwise: {ensemble_proba, individual_predictions, confidence,
                        agreement, thresholds, n_models}
        """
        if not self.is_available():
            return None

        if len(featured_df) < self.seq_length:
            logger.warning(
                f"Not enough rows for prediction: {len(featured_df)} < {self.seq_length}"
            )
            return None

        X_scaled = self._build_feature_matrix(featured_df)
        seq_window = X_scaled[-self.seq_length:]  # (seq_length, n_features)
        flat_row = X_scaled[-1:]                  # (1, n_features)

        individual: Dict[str, float] = {}

        with torch.no_grad():
            x_seq = torch.from_numpy(seq_window).float().unsqueeze(0)
            if self.lstm is not None:
                try:
                    p = self.lstm(x_seq)["direction_prob"].item()
                    individual["lstm"] = float(p)
                except Exception as e:
                    logger.warning(f"LSTM inference failed: {e}")
            if self.tft is not None:
                try:
                    p = self.tft(x_seq)["direction_prob"].item()
                    individual["transformer"] = float(p)
                except Exception as e:
                    logger.warning(f"Transformer inference failed: {e}")
            if self.cnn is not None:
                try:
                    p = self.cnn(x_seq)["direction_prob"].item()
                    individual["cnn"] = float(p)
                except Exception as e:
                    logger.warning(f"CNN inference failed: {e}")

        if self.xgb is not None:
            try:
                individual["xgboost"] = float(self.xgb.predict_proba(flat_row)[0, 1])
            except Exception as e:
                logger.warning(f"XGBoost inference failed: {e}")
        if self.lgb is not None:
            try:
                individual["lightgbm"] = float(self.lgb.predict_proba(flat_row)[0, 1])
            except Exception as e:
                logger.warning(f"LightGBM inference failed: {e}")
        if self.cat is not None:
            try:
                individual["catboost"] = float(self.cat.predict_proba(flat_row)[0, 1])
            except Exception as e:
                logger.warning(f"CatBoost inference failed: {e}")

        if len(individual) < 2:
            logger.warning(
                f"Only {len(individual)} model(s) produced predictions; ensemble disabled."
            )
            return None

        try:
            arr_dict = {k: np.array([v], dtype=np.float64) for k, v in individual.items()}
            proba = float(self.ensemble.predict_proba(arr_dict)[0])
        except Exception as e:
            logger.warning(f"Ensemble combine failed ({e}); using equal-weight mean.")
            proba = float(np.mean(list(individual.values())))

        proba = float(np.clip(proba, 1e-6, 1.0 - 1e-6))
        confidence = float(min(1.0, abs(proba - 0.5) * 2))

        up_votes = sum(1 for v in individual.values() if v > 0.5)
        agreement = up_votes / len(individual)

        return {
            "ensemble_proba": proba,
            "individual_predictions": {k: round(v, 4) for k, v in individual.items()},
            "confidence": confidence,
            "agreement": agreement,
            "thresholds": self.thresholds,
            "n_models": len(individual),
        }

    def predict_series(
        self,
        featured_df: pd.DataFrame,
        batch_size: int = 256,
    ) -> Optional[Dict]:
        """
        Batched inference for every timestep with at least `seq_length` history.

        Used by the backtest route so trained models drive signals at every step
        instead of the statistical fallback. Much faster than calling
        `predict_latest` row-by-row because deep models run in batches.

        Returns:
            None if inference is unavailable or not enough history.
            Otherwise:
                {
                    "ensemble_proba": np.ndarray (N - seq_length + 1,),
                    "individual_predictions": Dict[str, np.ndarray],
                    "valid_index_offset": int (= seq_length - 1),
                    "thresholds": Dict[str, float],
                    "n_models": int,
                }
        """
        if not self.is_available():
            return None
        if len(featured_df) < self.seq_length:
            logger.warning(
                f"predict_series: only {len(featured_df)} rows < seq_length {self.seq_length}"
            )
            return None

        X_scaled = self._build_feature_matrix(featured_df)
        n = X_scaled.shape[0]
        n_windows = n - self.seq_length + 1

        # Build all (seq_length) windows
        seq_arr = np.stack(
            [X_scaled[i : i + self.seq_length] for i in range(n_windows)],
            axis=0,
        )
        flat_arr = X_scaled[self.seq_length - 1 :]  # aligned: row at window's last step

        individual: Dict[str, np.ndarray] = {}

        def _deep_batched(model: torch.nn.Module) -> np.ndarray:
            preds = []
            with torch.no_grad():
                for start in range(0, n_windows, batch_size):
                    end = min(start + batch_size, n_windows)
                    x = torch.from_numpy(seq_arr[start:end]).float()
                    out = model(x)["direction_prob"].cpu().numpy()
                    preds.append(out)
            return np.concatenate(preds, axis=0)

        if self.lstm is not None:
            try:
                individual["lstm"] = _deep_batched(self.lstm)
            except Exception as e:
                logger.warning(f"LSTM batch inference failed: {e}")
        if self.tft is not None:
            try:
                individual["transformer"] = _deep_batched(self.tft)
            except Exception as e:
                logger.warning(f"Transformer batch inference failed: {e}")
        if self.cnn is not None:
            try:
                individual["cnn"] = _deep_batched(self.cnn)
            except Exception as e:
                logger.warning(f"CNN batch inference failed: {e}")

        if self.xgb is not None:
            try:
                individual["xgboost"] = self.xgb.predict_proba(flat_arr)[:, 1]
            except Exception as e:
                logger.warning(f"XGBoost batch inference failed: {e}")
        if self.lgb is not None:
            try:
                individual["lightgbm"] = self.lgb.predict_proba(flat_arr)[:, 1]
            except Exception as e:
                logger.warning(f"LightGBM batch inference failed: {e}")
        if self.cat is not None:
            try:
                individual["catboost"] = self.cat.predict_proba(flat_arr)[:, 1]
            except Exception as e:
                logger.warning(f"CatBoost batch inference failed: {e}")

        if len(individual) < 2:
            logger.warning(
                f"predict_series: only {len(individual)} model(s); ensemble disabled."
            )
            return None

        # Align lengths defensively (some models could return n_windows ± 0 rows)
        min_len = min(len(v) for v in individual.values())
        individual = {k: np.asarray(v[:min_len], dtype=np.float64)
                      for k, v in individual.items()}

        try:
            proba = self.ensemble.predict_proba(individual)
        except Exception as e:
            logger.warning(f"Ensemble batch combine failed ({e}); using mean.")
            proba = np.mean(np.column_stack(list(individual.values())), axis=1)

        proba = np.clip(proba, 1e-6, 1.0 - 1e-6)

        return {
            "ensemble_proba": proba.astype(np.float64),
            "individual_predictions": individual,
            "valid_index_offset": self.seq_length - 1,
            "thresholds": self.thresholds,
            "n_models": len(individual),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_inferencer: Optional[ModelInferencer] = None
_singleton_lock = threading.Lock()


def get_inferencer() -> ModelInferencer:
    """Return the process-wide ModelInferencer singleton."""
    global _inferencer
    if _inferencer is None:
        with _singleton_lock:
            if _inferencer is None:
                _inferencer = ModelInferencer()
    return _inferencer


def reload_inferencer() -> bool:
    """Reset the singleton (call after dropping new models in data/models/)."""
    global _inferencer
    with _singleton_lock:
        _inferencer = ModelInferencer()
        return _inferencer.is_available()
