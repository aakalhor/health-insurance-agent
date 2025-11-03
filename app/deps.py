from pathlib import Path
from functools import lru_cache
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "data"
MODELS_DIR = DATA_DIR / "models"
SQLITE_PATH = DATA_DIR / "plans.sqlite"

@lru_cache(maxsize=1)
def get_engine():
    if not SQLITE_PATH.exists():
        # Placeholder; the data pipeline will create this
        raise FileNotFoundError("plans.sqlite not found. Run data_pipeline/build_plans_sqlite.py")
    return create_engine(f"sqlite:///{SQLITE_PATH}")

@lru_cache(maxsize=1)
def get_model_paths():
    return {
        "p25": MODELS_DIR / "xgb_p25.json",
        "p50": MODELS_DIR / "xgb_p50.json",
        "p75": MODELS_DIR / "xgb_p75.json",
    }
