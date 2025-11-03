# app/services/llm_explainer.py
from __future__ import annotations
from typing import List, Optional, Any
from ..schemas import PlanResult, ProfileFeatures  # ok if you prefer to keep the type hints

SYSTEM_PROMPT = (
    "You are a helpful assistant explaining health insurance plan trade-offs. "
    "Use simple language. Do not invent numeric values; refer to Summary of Benefits and Coverage (SBC)."
)

_METAL_TRADEOFF = {
    "Bronze": "lowest premiums, highest cost when you use care; better if you expect low usage.",
    "Silver": "balanced premiums and costs; many people choose Silver for a middle-ground.",
    "Gold": "higher premiums, lower cost when you use care; better if you expect moderate usage.",
    "Platinum": "highest premiums, lowest cost when you use care; better if you expect high usage.",
    "Catastrophic": "very low premiums, very high cost until a large deductible; usually for under age 30 or hardship exemption.",
}

_NETWORK_HINT = {
    "HMO": "requires using in-network providers and may require referrals; typically lower premiums.",
    "EPO": "in-network only (similar to HMO) but usually no referrals; out-of-network not covered except emergencies.",
    "PPO": "more flexibility to see out-of-network providers, typically higher premiums.",
    "POS": "hybrid of HMO and PPO; check referral rules.",
    "Unknown": "check the SBC for network rules.",
}

def _g(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute or dict key with the same name."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _profile_rationale(p: ProfileFeatures) -> List[str]:
    notes: List[str] = []
    # Age
    try:
        age = int(_g(p, "age", 0))
        if age >= 30:
            notes.append("Age 30+ — Catastrophic plans may not be eligible unless you have a hardship exemption.")
        else:
            notes.append("Under age 30 — Catastrophic plans may be available if offered in your area.")
    except Exception:
        pass
    # Smoker
    smoker = bool(_g(p, "smoker", False))
    if smoker:
        notes.append("Smoker — premiums and expected spend tend to be higher.")
    else:
        notes.append("Non-smoker — premiums/expected spend tend to be lower than smoker profiles.")
    # BMI
    try:
        bmi = float(_g(p, "bmi", 0))
        if bmi >= 30:
            notes.append("Higher BMI often correlates with higher usage; richer metals (Gold/Platinum) can reduce point-of-care costs.")
        elif bmi < 23:
            notes.append("Lower expected usage; leaner metals (Bronze/Silver) can keep premiums down.")
    except Exception:
        pass
    # Children
    try:
        kids = int(_g(p, "children", 0))
        if kids > 0:
            notes.append("Dependents — predictable copays and broad pediatric networks can be valuable.")
    except Exception:
        pass
    # Network preference
    pref_net = _g(p, "preferences_network", None)
    if pref_net:
        notes.append(f"Network preference {str(pref_net).upper()} — prioritize matching network types.")
    return notes

def _plan_line(pl: PlanResult | dict) -> str:
    plan_name = str(_g(pl, "plan_name", "Unknown plan"))
    metal = str(_g(pl, "metal_tier", "Unknown"))
    premium = float(_g(pl, "premium_annual", 0.0))
    total = float(_g(pl, "total_cost_estimate", premium))
    # Prefer notes, fall back to network_type if present
    notes = _g(pl, "notes", None)
    if not notes:
        nt = _g(pl, "network_type", "Unknown")
        notes = f"Network: {nt}"
    hint = _METAL_TRADEOFF.get(metal, "check SBC for details.")
    return (
        f"- **{plan_name} ({metal})** — est. total ${total:,.0f} "
        f"(premium ≈ ${premium:,.0f}). {notes}. {hint}"
    )

def explain(profile: ProfileFeatures, plans: List[PlanResult] | List[dict], *, header: Optional[str] = None) -> str:
    hdr = header or "Top plan options for your profile"
    lines: List[str] = [hdr, ""]
    notes = _profile_rationale(profile)
    if notes:
        lines.append("**Why these might fit your situation:**")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    if not plans:
        lines.append("_No matching plans found. Try a different ZIP or relax filters (e.g., network/metal)._")
        return "\n".join(lines)
    lines.append("**Plans (review the SBC for deductibles, copays, and networks):**")
    for pl in plans:
        lines.append(_plan_line(pl))
    lines.append("")
    lines.append("**Tip:** Confirm deductibles, copays/coinsurance, and out-of-pocket maximum in the SBC and provider directory.")
    return "\n".join(lines)
