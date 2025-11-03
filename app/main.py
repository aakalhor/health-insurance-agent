# app/main.py
from __future__ import annotations

from typing import List, Optional
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Query
from fastapi import Path
from .schemas import PlanDetail
from .services.plan_lookup import fetch_plan_detail

from .schemas import (
    ProfileFeatures,
    SpendPrediction,
    RankResponse,
    ExplainRequest,   # kept for compatibility if you use it elsewhere
    ExplainResponse,  # kept for compatibility if you use it elsewhere
    # If you have PlanResult schema, we'll try to cast to it; otherwise we fall back to dicts
)
try:
    from .schemas import PlanResult  # optional
except Exception:
    PlanResult = None  # type: ignore

from .services.spend_model import QuantileSpendModel
from .services.ranker import rank_plans
from .services.llm_explainer import explain as explain_plans

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


app = FastAPI(title="Health Insurance Agent (Phase 3)")

# -----------------------------
# Singleton model
# -----------------------------
_model: Optional[QuantileSpendModel] = None

def get_model() -> QuantileSpendModel:
    global _model
    if _model is None:
        _model = QuantileSpendModel()
    return _model

# -----------------------------
# Utilities
# -----------------------------
def _clamp_quantiles(preds: dict) -> dict:
    """Ensure p25 ≤ p50 ≤ p75 (independent quantile models can cross)."""
    try:
        vals = sorted([float(preds["p25"]), float(preds["p50"]), float(preds["p75"])])
        preds["p25"], preds["p50"], preds["p75"] = vals
    except Exception:
        # if any key missing, just return as-is
        pass
    return preds

def _ns_profile_with_prefs(profile: ProfileFeatures, metal_tier_preference: Optional[List[str]]):
    """
    ranker expects attribute-style access (getattr). Build a SimpleNamespace
    from the pydantic model dump and attach optional preferences.
    """
    ns = SimpleNamespace(**profile.model_dump())
    if metal_tier_preference:
        ns.metal_tier_preference = metal_tier_preference
    return ns

def _format_band(preds: dict) -> str:
    return f"(estimated annual spend range: ${preds['p25']:,.0f}–${preds['p75']:,.0f}, median ${preds['p50']:,.0f})"

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "phase": 3}

@app.post("/predict", response_model=SpendPrediction)
def predict(profile: ProfileFeatures):
    try:
        preds = get_model().predict(profile.model_dump())
        preds = _clamp_quantiles(preds)
        return SpendPrediction(**preds)
    except Exception as e:
        raise HTTPException(status_code=501, detail=str(e))

@app.post("/rank", response_model=RankResponse)
def rank(
    profile: ProfileFeatures,
    top_k: int = Query(5, ge=1, le=50),
    metal_tier_preference: Optional[List[str]] = Query(
        default=None,
        description="Optional preference to prioritize certain metals (e.g., Silver,Gold)."
    ),
):
    # 1) predictions
    try:
        preds = get_model().predict(profile.model_dump())
        preds = _clamp_quantiles(preds)
    except Exception as e:
        raise HTTPException(status_code=501, detail=f"Predictor unavailable: {e}")

    # 2) rank (with optional preferences)
    prof_for_rank = _ns_profile_with_prefs(profile, metal_tier_preference)
    plans = rank_plans(prof_for_rank, preds, top_k=top_k)

    return RankResponse(predictions=preds, top_plans=plans)

@app.post("/explain")
def explain_endpoint(
    profile: ProfileFeatures,
    top_k: int = Query(5, ge=1, le=50),
    metal_tier_preference: Optional[List[str]] = Query(
        default=None,
        description="Optional preference to prioritize certain metals (e.g., Silver,Gold)."
    ),
):
    # 1) predict
    model = get_model()
    preds = model.predict(profile.model_dump())
    preds = _clamp_quantiles(preds)

    # 2) rank
    prof_for_rank = _ns_profile_with_prefs(profile, metal_tier_preference)
    top = rank_plans(prof_for_rank, preds, top_k=top_k)

    # 3) cast to PlanResult if available; otherwise, keep dicts
    plan_objs = top
    if PlanResult is not None:
        try:
            plan_objs = [PlanResult(**p) for p in top]  # type: ignore
        except Exception:
            plan_objs = top  # fall back to dicts

    # 4) explanation with quantile band in header
    header = f"Top plan options for your profile {_format_band(preds)}"
    explanation = explain_plans(profile, plan_objs, header=header)  # type: ignore

    # Keep the response simple (same shape you showed earlier)
    return {"explanation": explanation}

@app.post("/recommend")
def recommend(
    profile: ProfileFeatures,
    top_k: int = Query(5, ge=1, le=50),
    metal_tier_preference: Optional[List[str]] = Query(
        default=None,
        description="Optional preference to prioritize certain metals (e.g., Silver,Gold)."
    ),
):
    # 1) predict
    model = get_model()
    preds = model.predict(profile.model_dump())
    preds = _clamp_quantiles(preds)

    # 2) rank with preferences
    prof_for_rank = _ns_profile_with_prefs(profile, metal_tier_preference)
    top = rank_plans(prof_for_rank, preds, top_k=top_k)

    # 3) cast (optional)
    plan_objs = top
    if PlanResult is not None:
        try:
            plan_objs = [PlanResult(**p) for p in top]  # type: ignore
        except Exception:
            plan_objs = top

    # 4) explain with quantile band in header
    header = f"Top plan options for your profile {_format_band(preds)}"
    explanation = explain_plans(profile, plan_objs, header=header)  # type: ignore

    return {
        "predictions": preds,
        "top_plans": top,
        "explanation": explanation,
    }


@app.get("/plan/{plan_id}", response_model=PlanDetail)
def get_plan_detail(plan_id: str = Path(..., description="Exact PlanId variant, e.g., 40047MI0010001-00")):
    rec = fetch_plan_detail(plan_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found.")
    return PlanDetail(**rec)

# Serve / at our static index.html
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()
