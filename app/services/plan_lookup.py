from __future__ import annotations
import pandas as pd
from sqlalchemy import text
from ..deps import get_engine

SELECT_ONE = text("""
SELECT plan_id, plan_name, metal_tier, state, zip_code,
       premium_annual, network_type, hsa_eligible, dental_only,
       market_coverage, partial_county, sbc_url, plan_brochure
FROM plans
WHERE plan_id = :pid
LIMIT 1
""")

def fetch_plan_detail(plan_id: str) -> dict | None:
    eng = get_engine()
    df = pd.read_sql(SELECT_ONE, eng, params={"pid": plan_id})
    if df.empty:
        return None
    # Ensure numerics are plain Python types
    row = df.iloc[0].to_dict()
    row["premium_annual"] = float(row.get("premium_annual") or 0.0)
    for k in ("hsa_eligible", "dental_only", "partial_county"):
        if k in row and row[k] is not None:
            row[k] = int(row[k])
    return row
