# app/services/ranker.py
from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
from sqlalchemy import text

from ..deps import get_engine
from .validators import clamp_nonneg

# Only medical metal tiers (exclude dental "High/Low")
MEDICAL_METALS = {"Bronze", "Silver", "Gold", "Platinum", "Catastrophic"}


def _root(plan_id: str) -> str:
    """Strip variant suffix like '-00/-01/-02' from PlanId."""
    s = str(plan_id)
    return s.split("-")[0] if "-" in s else s


def _load_candidate_plans(profile) -> pd.DataFrame:
    """
    Load candidate plans for a given profile.

    Preference order:
      1) exact ZIP matches
      2) if none, fallback to same-state pool (includes rows with zip_code='00000' which
         your pipeline uses to mean broad/all-ZIP coverage when PartialCounty == 0)

    Apply source-level filters (Option B):
      - dental_only == 0
      - market_coverage == 'Individual'
      - metal_tier in MEDICAL_METALS
    """
    eng = get_engine()

    # 1) Exact ZIP pool
    q_zip = text(
        """
        SELECT plan_id, plan_name, metal_tier, premium_annual,
               network_type, hsa_eligible, state, zip_code,
               dental_only, market_coverage, partial_county
        FROM plans
        WHERE zip_code = :zip
        """
    )
    df = pd.read_sql(q_zip, eng, params={"zip": profile.zip_code})

    # 2) Fallback: state pool (includes possible '00000' rows for broad coverage)
    if df.empty:
        q_state_any = text(
            """
            SELECT plan_id, plan_name, metal_tier, premium_annual,
                   network_type, hsa_eligible, state, zip_code,
                   dental_only, market_coverage, partial_county
            FROM plans
            WHERE state = (SELECT state FROM plans WHERE zip_code = :zip LIMIT 1)
            """
        )
        df = pd.read_sql(q_state_any, eng, params={"zip": profile.zip_code})

    if df.empty:
        return df  # nothing to rank

    # ---- Source-level filters (Option B) ----
    if "dental_only" in df.columns:
        df = df[df["dental_only"] == 0]

    if "market_coverage" in df.columns:
        df = df[df["market_coverage"].astype(str).str.lower() == "individual"]

    if "metal_tier" in df.columns:
        df = df[df["metal_tier"].isin(MEDICAL_METALS)]

    # Normalize numeric premium
    if "premium_annual" in df.columns:
        df["premium_annual"] = pd.to_numeric(df["premium_annual"], errors="coerce").fillna(0.0)

    # Remove obviously bad premiums (<=0). If you want to keep them as last resort, comment this out.
    df = df[df["premium_annual"] > 0.0]

    return df


def rank_plans(profile, predictions: Dict[str, float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Combine predicted annual medical spend (use p50 by default) with premium_annual
    and return the best K plans by total cost estimate.
    """
    df = _load_candidate_plans(profile)
    if df.empty:
        return []

    # Age rule: exclude Catastrophic for age >= 30 (unless you add an override flag later)
    try:
        age_val = int(getattr(profile, "age", 0))
    except Exception:
        age_val = 0
    if age_val >= 30 and "metal_tier" in df.columns:
        df = df[df["metal_tier"] != "Catastrophic"]

    if df.empty:
        return []

    # De-duplicate plan variants (e.g., -00/-01/-02): keep the cheapest premium per root
    df["plan_id_root"] = df["plan_id"].map(_root)
    df = df.sort_values(["premium_annual", "plan_id"]).drop_duplicates(
        subset=["plan_id_root"], keep="first"
    )

    # Predicted median spend
    p50 = clamp_nonneg(predictions.get("p50", 0.0))

    # Total estimated annual cost = premium + predicted medical spend
    df["total_cost_estimate"] = df["premium_annual"].astype(float) + float(p50)

    # Sort by total cost, then premium; take top_k
    df = df.sort_values(["total_cost_estimate", "premium_annual"], ascending=[True, True]).head(top_k)

    # Build lightweight plan cards
    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        notes = f"Network: {r.get('network_type', 'Unknown')}"
        if str(r.get("zip_code", "")) == "00000":
            notes += ", Broad coverage"
        out.append(
            {
                "plan_id": r["plan_id"],
                "plan_name": r["plan_name"],
                "metal_tier": r["metal_tier"],
                "premium_annual": float(r["premium_annual"]),
                "total_cost_estimate": float(r["total_cost_estimate"]),
                "notes": notes,
            }
        )
    return out
