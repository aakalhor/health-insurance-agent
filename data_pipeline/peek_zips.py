# data_pipeline/peek_zips.py
import sqlite3
from pathlib import Path
DB = Path(__file__).resolve().parents[1] / "app" / "data" / "plans.sqlite"
with sqlite3.connect(DB) as conn:
    cur = conn.cursor()
    cur.execute("SELECT zip_code, COUNT(*) FROM plans GROUP BY zip_code ORDER BY COUNT(*) DESC LIMIT 20;")
    for z, c in cur.fetchall():
        print(z, c)
