"""
Build SQLite database for FastAPI
---------------------------------
Input:  cms_plans_clean.parquet
Output: app/data/plans.sqlite
"""

import pandas as pd
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "data"
CMS_PATH = DATA_DIR / "cms_plans_clean.parquet"
OUT_DB = DATA_DIR / "plans.sqlite"

def build_sqlite():
    if not CMS_PATH.exists():
        raise FileNotFoundError("Run prep_cms_pufs_2025.py first")

    df = pd.read_parquet(CMS_PATH)
    print(f"Loaded {len(df)} CMS plans")

    with sqlite3.connect(OUT_DB) as conn:
        df.to_sql("plans", conn, index=False, if_exists="replace")
        conn.commit()

    print(f"✅ SQLite database created at {OUT_DB}")

if __name__ == "__main__":
    build_sqlite()
