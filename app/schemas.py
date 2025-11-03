from pydantic import BaseModel, Field
from typing import List, Optional

class ProfileFeatures(BaseModel):
    age: int = Field(ge=0, le=120)
    zip_code: str
    income: Optional[float] = None
    smoker: Optional[bool] = None
    bmi: Optional[float] = None
    household_size: Optional[int] = 1
    chronic_conditions: Optional[List[str]] = []
    expected_visits_per_year: Optional[int] = 0
    preferences_network: Optional[str] = None   # e.g., HMO/EPO/PPO
    hsa_eligible_only: Optional[bool] = False
    metal_tier_preference: Optional[List[str]] = None  # e.g., ["Silver","Gold"]

class SpendPrediction(BaseModel):
    p25: float
    p50: float
    p75: float

class PlanResult(BaseModel):
    plan_id: str
    plan_name: str
    metal_tier: str
    premium_annual: float
    total_cost_estimate: float
    notes: Optional[str] = None

class RankResponse(BaseModel):
    predictions: SpendPrediction
    top_plans: List[PlanResult]

class ExplainRequest(BaseModel):
    profile: ProfileFeatures
    top_plans: List[PlanResult]

class ExplainResponse(BaseModel):
    text: str

from pydantic import BaseModel
from typing import Optional

class PlanDetail(BaseModel):
    plan_id: str
    plan_name: str
    metal_tier: str
    state: str
    zip_code: Optional[str] = None
    premium_annual: float
    network_type: str
    hsa_eligible: int
    dental_only: int
    market_coverage: str
    partial_county: int
    sbc_url: Optional[str] = None
    plan_brochure: Optional[str] = None
