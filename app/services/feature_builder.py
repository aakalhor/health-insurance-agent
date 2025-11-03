# app/services/feature_builder.py
from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from ..deps import DATA_DIR

# Map US state to Kaggle's 4 regions (approx.)
STATE_TO_REGION = {
    # Northeast
    "CT":"northeast","ME":"northeast","MA":"northeast","NH":"northeast","RI":"northeast","VT":"northeast",
    "NJ":"northeast","NY":"northeast","PA":"northeast",
    # Northwest (approximate to cover West/Northwest)
    "AK":"northwest","WA":"northwest","OR":"northwest","ID":"northwest","MT":"northwest","WY":"northwest",
    # Southeast
    "AL":"southeast","AR":"southeast","FL":"southeast","GA":"southeast","KY":"southeast","LA":"southeast",
    "MS":"southeast","NC":"southeast","SC":"southeast","TN":"southeast","VA":"southeast","WV":"southeast",
    # Southwest
    "AZ":"southwest","CA":"southwest","CO":"southwest","HI":"southwest","NM":"southwest","NV":"southwest",
    "OK":"southwest","TX":"southwest","UT":"southwest",
    # Assign rest sensibly
    "IL":"northeast","IN":"northeast","MI":"northeast","OH":"northeast","WI":"northeast",   # midwest→northeast approx
    "IA":"northwest","KS":"northwest","MN":"northwest","MO":"northwest","NE":"northwest","ND":"northwest","SD":"northwest",
    "DC":"northeast","MD":"northeast","DE":"northeast",
    "PR":"southeast",
}

ONE_HOT_REGIONS = ["northeast", "northwest", "southeast", "southwest"]

@dataclass
class Scaler:
    means: Dict[str, float]
    stds: Dict[str, float]

    def maybe_scale(self, name: str, value: float) -> float:
        if name in self.means and name in self.stds and self.stds[name] not in (0, None):
            return (value - self.means[name]) / (self.stds[name] + 1e-6)
        return value

def _load_scaler() -> Optional[Scaler]:
    """Optional scaler: if training saved stats.json, use it; else return None."""
    stats_path = DATA_DIR / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            obj = json.load(f)
        return Scaler(means=obj.get("means", {}), stds=obj.get("stds", {}))
    return None

def _infer_region(profile: Dict[str, Any]) -> str:
    # 1) explicit region if user passed (optional future field)
    explicit = (profile.get("region") or "").strip().lower()
    if explicit in ONE_HOT_REGIONS:
        return explicit
    # 2) try state code if available in profile (optional future)
    state = (profile.get("state") or "").strip().upper()
    if state and state in STATE_TO_REGION:
        return STATE_TO_REGION[state]
    # 3) fallback: infer by ZIP leading digit (very rough)
    z = (profile.get("zip_code") or "").strip()
    if z and z[0].isdigit():
        d = int(z[0])
        if d in (0,1,2,3): return "northeast"
        if d in (4,5):     return "southeast"
        if d in (6,7):     return "southwest"
        return "northwest"
    # default
    return "northeast"

def build_features(profile: Dict[str, Any], feature_order: list[str]) -> Dict[str, float]:
    """
    Map user profile → model feature vector (dict).
    Applies optional scaling if stats.json exists.
    """
    age = float(profile.get("age", 0))
    bmi = float(profile.get("bmi", 0))
    children = float(profile.get("expected_children", profile.get("children", 0)))
    sex = profile.get("sex")  # optional; if not provided, infer from nothing (default 0)
    if isinstance(sex, str):
        sex = 1.0 if sex.strip().lower() == "male" else 0.0
    elif sex is None:
        sex = 0.0
    else:
        sex = float(sex)

    smoker = 1.0 if bool(profile.get("smoker")) else 0.0

    region = _infer_region(profile)
    one_hot = {f"region_{r}": 1.0 if r == region else 0.0 for r in ONE_HOT_REGIONS}

    # Optional scaling (only if stats.json exists)
    scaler = _load_scaler()
    if scaler:
        age = scaler.maybe_scale("age", age)
        bmi = scaler.maybe_scale("bmi", bmi)
        children = scaler.maybe_scale("children", children)

    feats = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        **{ "region_northeast": one_hot["region_northeast"],
           "region_northwest": one_hot["region_northwest"],
           "region_southeast": one_hot["region_southeast"],
           "region_southwest": one_hot["region_southwest"] },
    }

    # Only keep what model expects (and fill missing with 0.0)
    return {f: float(feats.get(f, 0.0)) for f in feature_order}
