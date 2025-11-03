# app/services/spend_model.py
from typing import Dict, Any
import json, numpy as np
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from ..deps import DATA_DIR

MODELS_DIR = DATA_DIR / "models"

class QuantileSpendModel:
    def __init__(self):
        # load feature order + meta
        feats_path = DATA_DIR / "features.json"
        meta_path  = DATA_DIR / "meta.json"

        self.feature_order = []
        if feats_path.exists():
            with open(feats_path, "r") as f:
                self.feature_order = json.load(f)

        self.inv_transform = None
        self.framework = "xgboost"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("inverse_transform") == "expm1":
                self.inv_transform = np.expm1
            self.framework = meta.get("framework", "lightgbm")

        # load models
        self.models: Dict[str, Any] = {}
        if self.framework == "lightgbm":
            for q in ("p25","p50","p75"):
                path = MODELS_DIR / f"lgb_{q}.txt"
                if path.exists():
                    self.models[q] = lgb.Booster(model_file=str(path))
        else:
            # legacy xgb (not used now, but safe fallback)
            for q in ("p25","p50","p75"):
                path = MODELS_DIR / f"xgb_{q}.json"
                if path.exists():
                    booster = xgb.Booster()
                    booster.load_model(str(path))
                    self.models[q] = booster

    def _ensure_loaded(self):
        if not self.models or len(self.models) < 3:
            raise RuntimeError("Quantile models not found. Train with training/train_spend_model.py")
        if not self.feature_order:
            raise RuntimeError("features.json missing. Re-run training to export feature order.")

    def _predict_one(self, q: str, x_arr: np.ndarray, feature_names: list[str]) -> float:
        if self.framework == "lightgbm":
            yhat = float(self.models[q].predict(x_arr, num_iteration=self.models[q].best_iteration)[0])
        else:
            dm = xgb.DMatrix(x_arr, feature_names=feature_names)
            yhat = float(self.models[q].predict(dm)[0])
        if self.inv_transform:
            yhat = float(self.inv_transform(yhat))
        return max(0.0, yhat)

    def predict(self, profile: Dict[str, Any]) -> Dict[str, float]:
        from .feature_builder import build_features
        self._ensure_loaded()
        feats = build_features(profile, self.feature_order)
        x = np.array([[feats[f] for f in self.feature_order]])
        return {q: self._predict_one(q, x, self.feature_order) for q in ("p25","p50","p75")}
