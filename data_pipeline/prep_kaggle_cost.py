# data_pipeline/prep_kaggle_cost.py
"""
Prep Kaggle Medical Insurance Cost dataset (no normalization)
Input:  app/data/kaggle_insurance.csv
Output: app/data/kaggle_cost_prepared.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = DATA_DIR / "kaggle_insurance.csv"
OUT_PATH = DATA_DIR / "kaggle_cost_prepared.parquet"

def preprocess():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"{RAW_PATH} not found — please place Kaggle CSV there")

    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    # Encode categoricals
    df["sex"] = df["sex"].map({"male": 1, "female": 0})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
    df = pd.get_dummies(df, columns=["region"], prefix="region")

    # Clean numerics
    for col in ["age", "bmi", "children", "charges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["charges"] = df["charges"].clip(lower=0)
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df["children"] = df["children"].fillna(0).astype(int)
    df["age"] = df["age"].fillna(df["age"].median()).astype(int)

    # Ensure all expected region columns exist
    for c in ["region_northeast","region_northwest","region_southeast","region_southwest"]:
        if c not in df.columns:
            df[c] = 0

    df.to_parquet(OUT_PATH, index=False)
    print(f"✅ Saved cleaned dataset (no normalization) → {OUT_PATH}")

if __name__ == "__main__":
    preprocess()
