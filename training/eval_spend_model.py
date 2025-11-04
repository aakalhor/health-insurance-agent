# training/eval_spend_model.py
from __future__ import annotations

import argparse
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

FEATURES_PATH = DATA_DIR / "features.json"
META_PATH = DATA_DIR / "meta.json"

CSV_PATH = DATA_DIR / "kaggle_insurance.csv"
PARQUET_PATH = DATA_DIR / "kaggle_cost_prepared.parquet"


# --------- Utils ---------
def inv_expm1(a: np.ndarray) -> np.ndarray:
    return np.expm1(a)

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Quantile (pinball) loss in the same space as y_true/y_pred (log space here)."""
    e = y_true - y_pred
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))

def _ensure_region_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure region_* dummies exist if 'region' column is available."""
    if "region" in df.columns:
        d = pd.get_dummies(df["region"].astype(str), prefix="region")
        for want in ["region_northeast", "region_northwest", "region_southeast", "region_southwest"]:
            if want not in d.columns:
                d[want] = 0
        df = pd.concat(
            [df.drop(columns=["region"]), d[["region_northeast", "region_northwest", "region_southeast", "region_southwest"]]],
            axis=1
        )
    return df

def _coerce_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """Match the training feature encodings."""
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

def _load_dataset() -> pd.DataFrame:
    """Load evaluation dataset; expects a 'charges' column."""
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    raise FileNotFoundError(
        f"Missing dataset: {PARQUET_PATH} or {CSV_PATH} not found."
    )

def _build_Xy(df: pd.DataFrame, features: list[str], target_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if "charges" not in df.columns:
        raise ValueError("Evaluation data must contain a 'charges' column.")
    df = _ensure_region_dummies(df.copy())
    df = _coerce_feature_types(df)
    for col in features:
        if col not in df.columns:
            df[col] = 0
    X = df[features].values.astype(float)
    if target_name != "log1p_charges":
        raise ValueError(f"Unsupported target in meta.json: {target_name}")
    y_log = np.log1p(pd.to_numeric(df["charges"], errors="coerce").fillna(0).values.astype(float))
    return X, y_log

def _load_booster(tag: str, model_dir: Path) -> lgb.Booster:
    path = model_dir / f"lgb_{tag}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return lgb.Booster(model_file=str(path))

def _metrics_dollars(y_log: np.ndarray, yhat_log: np.ndarray, tau: float) -> Dict[str, float]:
    pin = pinball_loss(y_log, yhat_log, tau)
    y = inv_expm1(y_log); yhat = inv_expm1(yhat_log)
    mae = mean_absolute_error(y, yhat)
    rmse = math.sqrt(mean_squared_error(y, yhat))
    r2 = r2_score(y, yhat)
    return {
        "pinball_loss_log": pin,
        "mae_dollars": mae,
        "rmse_dollars": rmse,
        "r2": r2,
    }

def _crossing_rate(p25: np.ndarray, p50: np.ndarray, p75: np.ndarray) -> float:
    """Share of rows where quantiles cross: p25>p50 or p50>p75."""
    bad = np.sum((p25 > p50) | (p50 > p75))
    return float(bad) / float(len(p25))


# --------- CLI / Main ---------
def main():
    ap = argparse.ArgumentParser(description="Evaluate quantile LightGBM models on a holdout test split.")
    ap.add_argument("--model_dir", default=str(MODEL_DIR), help="Directory with lgb_p25.txt, lgb_p50.txt, lgb_p75.txt")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split fraction (default 0.2)")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for the test split")
    ap.add_argument("--out", default=str(MODEL_DIR / "metrics_eval.json"), help="Where to write the JSON metrics")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)

    # Load spec + data and build X/y
    features, meta = _load_feature_spec()
    df = _load_dataset()
    X, y_log = _build_Xy(df, features, meta.get("target", "log1p_charges"))

    # Make a test split (independent from the training script's val split)
    X_tr, X_te, y_tr_log, y_te_log = train_test_split(
        X, y_log, test_size=args.test_size, random_state=args.seed
    )
    del X_tr, y_tr_log  # we only need the test set here

    # Load boosters
    m25 = _load_booster("p25", model_dir)
    m50 = _load_booster("p50", model_dir)
    m75 = _load_booster("p75", model_dir)

    # Predict in log space
    te25 = m25.predict(X_te, num_iteration=m25.best_iteration)
    te50 = m50.predict(X_te, num_iteration=m50.best_iteration)
    te75 = m75.predict(X_te, num_iteration=m75.best_iteration)

    # Metrics per quantile
    metrics = {
        "p25": _metrics_dollars(y_te_log, te25, 0.25),
        "p50": _metrics_dollars(y_te_log, te50, 0.50),
        "p75": _metrics_dollars(y_te_log, te75, 0.75),
        "quantile_crossing_rate": _crossing_rate(te25, te50, te75),
        "n_test": int(len(y_te_log)),
        "seed": args.seed,
        "test_size": args.test_size,
    }

    # Print a concise summary
    for tag, tau in [("p25", 0.25), ("p50", 0.50), ("p75", 0.75)]:
        m = metrics[tag]
        print(f"[{tag}] pinball(log): {m['pinball_loss_log']:.5f} | "
              f"MAE: ${m['mae_dollars']:,.2f} | RMSE: ${m['rmse_dollars']:,.2f} | R²: {m['r2']:.4f}")
    print(f"Quantile crossing rate: {metrics['quantile_crossing_rate']:.3%} on N={metrics['n_test']}")

    # Save JSON
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved evaluation metrics → {out_path}")

if __name__ == "__main__":
    main()
