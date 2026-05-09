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


class TreeModels:
    """
    Wrapper for tree-based gradient boosting models.
    Trains XGBoost, LightGBM, and CatBoost with unified interface.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.feature_importances: Dict[str, np.ndarray] = {}

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Train XGBoost classifier."""
        import xgboost as xgb

        default_params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 5,
            "gamma": 0.1,
            "scale_pos_weight": 1.0,
            "tree_method": "hist",
            "device": "cuda",
            "random_state": 42,
        }
        if params:
            default_params.update(params)

        n_estimators = default_params.pop("n_estimators", 500)

        model = xgb.XGBClassifier(n_estimators=n_estimators, **default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        self.models["xgboost"] = model
        self.feature_importances["xgboost"] = model.feature_importances_

        # Evaluate
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
        """Train LightGBM classifier."""
        import lightgbm as lgb

        default_params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": "gbdt",
            "max_depth": 7,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
            "device": "gpu",
            "verbose": -1,
            "random_state": 42,
        }
        if params:
            default_params.update(params)

        model = lgb.LGBMClassifier(**default_params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(50)],
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
        """Train CatBoost classifier."""
        from catboost import CatBoostClassifier

        default_params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 50,
            "task_type": "GPU",
            "eval_metric": "AUC",
            "auto_class_weights": "Balanced",
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
        """Get predictions from all trained tree models."""
        predictions = {}
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X)[:, 1]
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
