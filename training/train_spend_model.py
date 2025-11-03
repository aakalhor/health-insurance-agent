import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "data"
MODELS_DIR = DATA_DIR / "models"
PARQUET_PATH = DATA_DIR / "kaggle_cost_prepared.parquet"
FEATURES_PATH = DATA_DIR / "features.json"
META_PATH = DATA_DIR / "meta.json"

SEED = 42
N_ROUNDS = 1500

def load_data():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Missing {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    if "charges" not in df.columns:
        raise KeyError("Expected 'charges' in Kaggle data.")
    # log1p target for stability (as before)
    y = np.log1p(df["charges"].astype(float))
    X = df.drop(columns=["charges"])
    return X, y

def train_quantile_lgbm(X_tr, y_tr, X_va, y_va, alpha: float) -> lgb.Booster:
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=list(X_tr.columns))
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=list(X_va.columns))
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "seed": 42,
        "verbose": -1,
    }
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),  # silence training logs
    ]
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=callbacks,          # <- use callbacks instead of early_stopping_rounds
    )
    return model

def evaluate(model: lgb.Booster, X_va: pd.DataFrame, y_va: np.ndarray, tag: str):
    yhat_log = model.predict(X_va, num_iteration=model.best_iteration)
    y_true = np.expm1(y_va)
    y_pred = np.expm1(yhat_log)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"[{tag}] MAE: ${mae:,.2f} | R²: {r2:.4f}")
    return mae, r2

def main():
    X, y = load_data()
    feature_cols = list(X.columns)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = {}
    for name, alpha in [("p25", 0.25), ("p50", 0.50), ("p75", 0.75)]:
        print(f"Training LightGBM quantile {name} (alpha={alpha}) ...")
        model = train_quantile_lgbm(X_tr, y_tr, X_va, y_va, alpha)
        out_path = MODELS_DIR / f"lgb_{name}.txt"
        model.save_model(str(out_path))
        print(f"  → saved {out_path}")
        evaluate(model, X_va, y_va, tag=name)
        models[name] = out_path.name

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(META_PATH, "w") as f:
        json.dump(
            {
                "target": "log1p_charges",
                "inverse_transform": "expm1",
                "features": feature_cols,
                "quantiles": {"p25": 0.25, "p50": 0.50, "p75": 0.75},
                "framework": "lightgbm",
            },
            f,
            indent=2,
        )
    print("✅ Training complete (LightGBM).")

if __name__ == "__main__":
    main()
