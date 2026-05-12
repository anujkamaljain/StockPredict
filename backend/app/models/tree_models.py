"""
Tree-based models: XGBoost, LightGBM, CatBoost.
These are the workhorses for tabular financial data — often outperform DL on structured features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
import joblib
from pathlib import Path


def _detect_gpu() -> Dict[str, bool]:
    """
    Detect GPU availability for each tree library.
    Safe to call even if torch/lightgbm/catboost are not installed.
    """
    gpu = {"xgboost": False, "lightgbm": False, "catboost": False}

    # Check CUDA via torch (covers XGBoost and CatBoost)
    try:
        import torch
        if torch.cuda.is_available():
            gpu["xgboost"] = True
            gpu["catboost"] = True
    except (ImportError, Exception):
        pass

    # LightGBM GPU requires OpenCL — probe only if lightgbm is installed
    try:
        import lightgbm as _lgb
        _clf = _lgb.LGBMClassifier(n_estimators=1, device="gpu", verbose=-1)
        _clf.fit(np.zeros((4, 1)), np.array([0, 1, 0, 1]))
        gpu["lightgbm"] = True
    except ImportError:
        pass  # lightgbm not installed — skip silently
    except Exception:
        pass  # installed but GPU not available — CPU fallback

    return gpu


class TreeModels:
    """
    Wrapper for tree-based gradient boosting models.
    Trains XGBoost, LightGBM, and CatBoost with unified interface.
    Auto-detects GPU availability and falls back to CPU gracefully.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.feature_importances: Dict[str, np.ndarray] = {}
        self._gpu = _detect_gpu()
        logger.info(f"Tree model GPU support: {self._gpu}")

    @staticmethod
    def _ensure_int_labels(y: np.ndarray) -> np.ndarray:
        """Convert {0,1}-valued labels to int regardless of original dtype."""
        return np.asarray(y).astype(int)

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Train XGBoost classifier with early stopping and class-balance handling."""
        import xgboost as xgb

        y_train = self._ensure_int_labels(y_train)
        y_val = self._ensure_int_labels(y_val)

        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        scale_pos = neg_count / max(pos_count, 1)

        default_params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 1500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 5,
            "gamma": 0.1,
            "scale_pos_weight": float(np.clip(scale_pos, 0.5, 2.0)),
            "tree_method": "hist",
            "device": "cuda" if self._gpu["xgboost"] else "cpu",
            "random_state": 42,
            "early_stopping_rounds": 75,
        }
        if params:
            default_params.update(params)

        n_estimators = default_params.pop("n_estimators", 1500)
        early_stopping = default_params.pop("early_stopping_rounds", 75)

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping,
            **default_params,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        self.models["xgboost"] = model
        self.feature_importances["xgboost"] = model.feature_importances_

        train_pred = model.predict_proba(X_train)[:, 1]
        val_pred = model.predict_proba(X_val)[:, 1]

        metrics = self._compute_metrics(y_train, train_pred, y_val, val_pred)
        logger.info(f"XGBoost trained: val_auc={metrics['val_auc']:.4f}")

        if self.model_dir:
            model.save_model(str(self.model_dir / "xgboost_model.json"))

        return metrics

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Train LightGBM classifier with early stopping."""
        import lightgbm as lgb

        y_train = self._ensure_int_labels(y_train)
        y_val = self._ensure_int_labels(y_val)

        default_params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": "gbdt",
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 1500,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
            "is_unbalance": True,
            "device": "gpu" if self._gpu["lightgbm"] else "cpu",
            "verbose": -1,
            "random_state": 42,
        }
        if params:
            default_params.update(params)

        model = lgb.LGBMClassifier(**default_params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(75, verbose=True),
            ],
        )

        self.models["lightgbm"] = model
        self.feature_importances["lightgbm"] = model.feature_importances_

        train_pred = model.predict_proba(X_train)[:, 1]
        val_pred = model.predict_proba(X_val)[:, 1]

        metrics = self._compute_metrics(y_train, train_pred, y_val, val_pred)
        logger.info(f"LightGBM trained: val_auc={metrics['val_auc']:.4f}")

        if self.model_dir:
            joblib.dump(model, self.model_dir / "lightgbm_model.pkl")

        return metrics

    def train_catboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Train CatBoost classifier with early stopping."""
        from catboost import CatBoostClassifier

        y_train = self._ensure_int_labels(y_train)
        y_val = self._ensure_int_labels(y_val)

        default_params = {
            "iterations": 1500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 100,
            "task_type": "GPU" if self._gpu["catboost"] else "CPU",
            "eval_metric": "AUC",
            "auto_class_weights": "Balanced",
            "early_stopping_rounds": 75,
        }
        if params:
            default_params.update(params)

        model = CatBoostClassifier(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )

        self.models["catboost"] = model
        self.feature_importances["catboost"] = model.feature_importances_

        train_pred = model.predict_proba(X_train)[:, 1]
        val_pred = model.predict_proba(X_val)[:, 1]

        metrics = self._compute_metrics(y_train, train_pred, y_val, val_pred)
        logger.info(f"CatBoost trained: val_auc={metrics['val_auc']:.4f}")

        if self.model_dir:
            model.save_model(str(self.model_dir / "catboost_model.cbm"))

        return metrics

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all trained tree models with robust error handling."""
        predictions = {}
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X)[:, 1]
                proba = np.nan_to_num(np.asarray(proba, dtype=np.float64),
                                      nan=0.5, posinf=1.0, neginf=0.0)
                proba = np.clip(proba, 1e-6, 1.0 - 1e-6)
                predictions[name] = proba
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")
        return predictions

    def get_feature_importance(
        self, feature_names: Optional[List[str]] = None, top_n: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """Get top feature importances from each model."""
        result = {}
        for name, importance in self.feature_importances.items():
            df = pd.DataFrame({
                "feature": feature_names if feature_names else [f"f_{i}" for i in range(len(importance))],
                "importance": importance,
            })
            df = df.sort_values("importance", ascending=False).head(top_n)
            result[name] = df
        return result

    @staticmethod
    def _compute_metrics(
        y_train: np.ndarray,
        train_pred: np.ndarray,
        y_val: np.ndarray,
        val_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

        train_class = (train_pred > 0.5).astype(int)
        val_class = (val_pred > 0.5).astype(int)

        return {
            "train_auc": float(roc_auc_score(y_train, train_pred)),
            "val_auc": float(roc_auc_score(y_val, val_pred)),
            "train_accuracy": float(accuracy_score(y_train, train_class)),
            "val_accuracy": float(accuracy_score(y_val, val_class)),
            "train_logloss": float(log_loss(y_train, train_pred)),
            "val_logloss": float(log_loss(y_val, val_pred)),
        }
