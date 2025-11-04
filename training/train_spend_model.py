# training/train_spend_model.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------- Paths ---------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "data"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_DIR / "features.json"  
META_PATH = DATA_DIR / "meta.json"          

CSV_PATH = DATA_DIR / "kaggle_insurance.csv"
PARQUET_PATH = DATA_DIR / "kaggle_cost_prepared.parquet"


# --------- Utils ---------
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Quantile (pinball) loss in the same space as y_true/y_pred (log space here)."""
    e = y_true - y_pred
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))

def inv_expm1(a: np.ndarray) -> np.ndarray:
    return np.expm1(a)

def _ensure_region_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Create region_{northeast,northwest,southeast,southwest} dummies if 'region' exists."""
    if "region" in df.columns:
        d = pd.get_dummies(df["region"].astype(str), prefix="region")
        for want in ["region_northeast", "region_northwest", "region_southeast", "region_southwest"]:
            if want not in d.columns:
                d[want] = 0
        df = pd.concat([df.drop(columns=["region"]), d[["region_northeast","region_northwest","region_southeast","region_southwest"]]], axis=1)
    return df

def _coerce_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    # Standard Kaggle insurance types
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).map({"male": 1, "female": 0}).fillna(0).astype(int)
    if "smoker" in df.columns:
        df["smoker"] = df["smoker"].astype(str).map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}).fillna(0).astype(int)
    for col in ["age", "bmi", "children"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def _load_feature_spec() -> Tuple[list[str], dict]:
    with open(FEATURES_PATH, "r") as f:
        feats = json.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return feats, meta

def _load_training_frame() -> pd.DataFrame:
    """Load training data from parquet or csv; expect 'charges' present."""
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        raise FileNotFoundError(
            f"Missing dataset: {PARQUET_PATH} or {CSV_PATH} not found."
        )
    return df

def _build_Xy(df: pd.DataFrame, features: list[str], target_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X (ordered by features.json) and y=log1p(charges) as specified in meta.json.
    Expects original 'charges' column in df.
    """
    if "charges" not in df.columns:
        raise ValueError("Training data must contain a 'charges' column.")

    df = _ensure_region_dummies(df.copy())
    df = _coerce_feature_types(df)

    # Make sure all features exist; add 0-filled if missing
    for col in features:
        if col not in df.columns:
            df[col] = 0

    X = df[features].values.astype(float)
    if target_name == "log1p_charges":
        y_log = np.log1p(pd.to_numeric(df["charges"], errors="coerce").fillna(0).values.astype(float))
    else:
        raise ValueError(f"Unsupported target in meta.json: {target_name}")

    return X, y_log


# --------- LightGBM training ---------
def fit_quantile_lgbm(
    X_tr: np.ndarray,
    y_tr_log: np.ndarray,
    X_va: np.ndarray,
    y_va_log: np.ndarray,
    tau: float,
    num_boost_round: int = 4000,
    early_stopping_rounds: int = 200,
) -> tuple[lgb.Booster, Dict]:
    params = {
        "objective": "quantile",
        "alpha": tau,
        "metric": "quantile",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_depth": -1,
        "verbosity": -1,
        "seed": 42,
    }

    dtrain = lgb.Dataset(X_tr, label=y_tr_log)
    dvalid = lgb.Dataset(X_va, label=y_va_log, reference=dtrain)

    evals_result: Dict = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=50),
        lgb.record_evaluation(evals_result),
    ]

    print(f"\nTraining LightGBM quantile tau={tau} ...")
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    return model, evals_result

def save_learning_curve(evals_result: Dict, tau: float):
    train_curve = evals_result.get("train", {}).get("quantile", [])
    valid_curve = evals_result.get("valid", {}).get("quantile", [])
    iters = list(range(1, max(len(train_curve), len(valid_curve)) + 1))
    df = pd.DataFrame({"iteration": iters})
    if train_curve:
        df["train_quantile"] = pd.Series(train_curve)
    if valid_curve:
        df["valid_quantile"] = pd.Series(valid_curve)
    out = MODEL_DIR / f"training_curves_lgb_{int(tau*100)}.csv"
    df.to_csv(out, index=False)
    print(f"  → saved learning curve: {out}")

def eval_in_dollars(model: lgb.Booster, X: np.ndarray, y_log: np.ndarray, tau: float) -> Dict[str, float]:
    yhat_log = model.predict(X, num_iteration=model.best_iteration)
    pin = pinball_loss(y_log, yhat_log, tau)
    y = inv_expm1(y_log)
    yhat = inv_expm1(yhat_log)
    mae = mean_absolute_error(y, yhat)
    rmse = math.sqrt(mean_squared_error(y, yhat))
    r2 = r2_score(y, yhat)
    return {
        "pinball_loss_log": pin,
        "mae_dollars": mae,
        "rmse_dollars": rmse,
        "r2": r2,
    }


# --------- Main ---------
def main():
    # 1) Load spec & data
    features, meta = _load_feature_spec()
    target_name = meta.get("target", "log1p_charges")
    df = _load_training_frame()

    # 2) Build matrices and split
    X, y_log = _build_Xy(df, features, target_name)
    X_tr, X_va, y_tr_log, y_va_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # 3) Train three quantiles
    results_summary: Dict[str, Dict[str, float]] = {}
    for tau, tag in [(0.25, "p25"), (0.50, "p50"), (0.75, "p75")]:
        model, ev = fit_quantile_lgbm(X_tr, y_tr_log, X_va, y_va_log, tau)
        save_learning_curve(ev, tau)

        # Save model
        model_path = MODEL_DIR / f"lgb_{tag}.txt"
        model.save_model(str(model_path))
        print(f"  → saved {model_path}")

        # 4) Report & store validation metrics (in dollars)
        m = eval_in_dollars(model, X_va, y_va_log, tau)
        print(f"[{tag}] pinball(log): {m['pinball_loss_log']:.5f} | "
              f"MAE: ${m['mae_dollars']:,.2f} | RMSE: ${m['rmse_dollars']:,.2f} | R²: {m['r2']:.4f}")
        results_summary[tag] = m

    # 5) Save summary JSON
    with open(MODEL_DIR / "metrics_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\n✅ Training complete. Curves + metrics saved to app/data/models/")

if __name__ == "__main__":
    main()
