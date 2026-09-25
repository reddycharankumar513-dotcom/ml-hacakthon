"""
Gradient Boosting Matcher Model Wrapper (LightGBM / XGBoost).
"""

import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from .utils import setup_logging, Timer

logger = setup_logging()

EXCLUDE_COLUMNS = [
    "source1_entity_id", "candidate_entity_id", "label", "blocker_names",
    "is_hard_negative", "s1_id", "candidate_id"
]

class EntityResolutionMatcher:
    """
    GBDT binary classification matcher model.
    Predicts probability P(match | candidate_pair).
    """
    def __init__(self, model_type: str = "lightgbm", params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type.lower()
        self.params = params or {}
        self.model = None
        self.feature_names = []

    def fit(self, df_train: pd.DataFrame, df_val: Optional[pd.DataFrame] = None):
        """Train GBDT model on pair features."""
        self.feature_names = [c for c in df_train.columns if c not in EXCLUDE_COLUMNS]
        logger.info(f"Training {self.model_type.upper()} model on {len(df_train):,} pairs with {len(self.feature_names)} features...")
        
        X_train = df_train[self.feature_names].values.astype(np.float32)
        y_train = df_train["label"].values.astype(int)
        
        X_val, y_val = None, None
        if df_val is not None and not df_val.empty:
            X_val = df_val[self.feature_names].values.astype(np.float32)
            y_val = df_val["label"].values.astype(int)

        with Timer(f"Training {self.model_type.upper()}"):
            if self.model_type == "lightgbm":
                lgb_params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",
                    "n_estimators": self.params.get("n_estimators", 300),
                    "learning_rate": self.params.get("learning_rate", 0.05),
                    "num_leaves": self.params.get("num_leaves", 63),
                    "max_depth": self.params.get("max_depth", 8),
                    "subsample": self.params.get("subsample", 0.8),
                    "colsample_bytree": self.params.get("colsample_bytree", 0.8),
                    "class_weight": self.params.get("class_weight", "balanced"),
                    "random_state": self.params.get("random_seed", 42),
                    "n_jobs": self.params.get("n_jobs", -1),
                    "verbose": -1
                }
                
                self.model = lgb.LGBMClassifier(**lgb_params)
                if X_val is not None:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                    )
                else:
                    self.model.fit(X_train, y_train)
                    
            elif self.model_type == "xgboost":
                pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
                xgb_params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "n_estimators": self.params.get("n_estimators", 300),
                    "learning_rate": self.params.get("learning_rate", 0.05),
                    "max_depth": self.params.get("max_depth", 8),
                    "subsample": self.params.get("subsample", 0.8),
                    "colsample_bytree": self.params.get("colsample_bytree", 0.8),
                    "scale_pos_weight": pos_weight,
                    "random_state": self.params.get("random_seed", 42),
                    "n_jobs": self.params.get("n_jobs", -1)
                }
                self.model = xgb.XGBClassifier(**xgb_params)
                if X_val is not None:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:
                    self.model.fit(X_train, y_train)

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Predict pair match probabilities."""
        X = df_features[self.feature_names].values.astype(np.float32)
        probas = self.model.predict_proba(X)[:, 1]
        return probas

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance DataFrame."""
        if not self.model or not self.feature_names:
            return pd.DataFrame()
            
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            df_imp = pd.DataFrame({
                "feature": self.feature_names,
                "importance": imp
            }).sort_values("importance", ascending=False)
            return df_imp
        return pd.DataFrame()

    def save(self, filepath: str):
        """Save model checkpoint to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names, "type": self.model_type}, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model checkpoint from disk."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.model_type = data["type"]
        logger.info(f"Loaded model from {filepath}")
