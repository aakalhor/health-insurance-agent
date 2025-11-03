"""
Prep CMS Exchange PUFs 2025 (Option B: carry medical/dental + market coverage + partial county)

Inputs in app/data/:
  - Plan_Attributes_PUF.csv
  - Rate_PUF.csv
  - Service_Area_PUF.csv

Output:
  - app/data/cms_plans_clean.parquet
"""

from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PLAN_PATH = DATA_DIR / "Plan_Attributes_PUF.csv"
RATE_PATH = DATA_DIR / "Rate_PUF.csv"
SERVICE_PATH = DATA_DIR / "Service_Area_PUF.csv"
OUT_PATH = DATA_DIR / "cms_plans_clean.parquet"


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _split_zipcodes(zips: str) -> list[str]:
    if not isinstance(zips, str) or not zips.strip():
        return []
    s = re.sub(r"[;\s]+", ",", zips.strip())
    return [z for z in s.split(",") if z]


def preprocess():
    plan_df = _load(PLAN_PATH)
    rate_df = _load(RATE_PATH)
    svc_df  = _load(SERVICE_PATH)

    # -------------------------
    # Plan Attributes
    # -------------------------
    plan_df = plan_df.rename(columns={
        "planid": "plan_id",
        "planmarketingname": "plan_name",
        "metallevel": "metal_tier",
        "statecode": "state",
        "serviceareaid": "service_area_id",
        "plantype": "network_type",
        "ishsaeligible": "hsa_eligible",
        "dentalonlyplan": "dental_only",
        "marketcoverage": "market_coverage",
        "urlforsummaryofbenefitscoverage": "sbc_url",
        "planbrochure": "plan_brochure",
    })

    # Normalize flags
    plan_df["hsa_eligible"] = (
        plan_df.get("hsa_eligible", "0").astype(str).str.lower()
        .map({"yes": 1, "true": 1, "1": 1, "y": 1, "no": 0, "false": 0, "0": 0, "n": 0})
        .fillna(0).astype(int)
    )
    plan_df["dental_only"] = (
        plan_df.get("dental_only", "0").astype(str).str.lower()
        .map({"yes": 1, "true": 1, "1": 1, "y": 1, "no": 0, "false": 0, "0": 0, "n": 0})
        .fillna(0).astype(int)
    )
    plan_df["market_coverage"] = plan_df.get("market_coverage", "Unknown").astype(str)

    plan_df = plan_df[[
        "plan_id", "plan_name", "metal_tier", "state", "service_area_id",
        "network_type", "hsa_eligible", "dental_only", "market_coverage", "sbc_url", "plan_brochure"
    ]].dropna(subset=["plan_id", "plan_name", "metal_tier", "state"])

    # Build root id (strip variant suffix like -00/-01/-02)
    def _root(x: str) -> str:
        x = str(x)
        return x.split("-")[0] if "-" in x else x

    plan_df["plan_id_root"] = plan_df["plan_id"].map(_root)

    # -------------------------
    # Rate PUF (robust reference; monthly -> annual), joined by plan_id_root
    # -------------------------
    rate_df = rate_df.rename(columns={
        "planid": "plan_id",
        "tobacco": "tobacco",
        "age": "age",
        "individualrate": "individual_rate",
    })
    # keep only rows with a numeric rate
    rate_df["individual_rate"] = pd.to_numeric(rate_df["individual_rate"], errors="coerce")
    rate_df = rate_df.dropna(subset=["individual_rate"])
    # normalize fields used for filters
    rate_df["age"] = rate_df["age"].astype(str)
    rate_df["tobacco"] = rate_df["tobacco"].astype(str).str.lower()

    # root id for join
    rate_df["plan_id_root"] = rate_df["plan_id"].map(_root)

    def _pick_ref(df: pd.DataFrame) -> pd.DataFrame:
        # 1) 27 & nontobacco
        s1 = df[(df["age"] == "27") & (df["tobacco"] == "nontobacco")]
        if not s1.empty:
            r = s1.groupby("plan_id_root")["individual_rate"].median().rename("monthly_ref")
            return r.reset_index()
        # 2) 40 & nontobacco (fallback)
        s2 = df[(df["age"] == "40") & (df["tobacco"] == "nontobacco")]
        if not s2.empty:
            r = s2.groupby("plan_id_root")["individual_rate"].median().rename("monthly_ref")
            return r.reset_index()
        # 3) any nontobacco (median across ages)
        s3 = df[df["tobacco"] == "nontobacco"]
        if not s3.empty:
            r = s3.groupby("plan_id_root")["individual_rate"].median().rename("monthly_ref")
            return r.reset_index()
        # 4) any row (median across ages & tobacco)
        r = df.groupby("plan_id_root")["individual_rate"].median().rename("monthly_ref")
        return r.reset_index()

    ref_rate = _pick_ref(rate_df)
    ref_rate["premium_annual"] = (ref_rate["monthly_ref"] * 12).astype(float)
    ref_rate = ref_rate[["plan_id_root", "premium_annual"]].drop_duplicates("plan_id_root")

    # -------------------------
    # Service Area (explode ZIPs, carry PartialCounty, keep '00000' when partial_county==0)
    # -------------------------
    svc_df = svc_df.rename(columns={
        "serviceareaid": "service_area_id",
        "statecode": "state",
        "zipcodes": "zip_codes",
        "partialcounty": "partial_county",
    })
    # Normalize partial_county to 0/1
    svc_df["partial_county"] = (
        svc_df.get("partial_county", "No").astype(str).str.lower()
        .map({"yes": 1, "true": 1, "1": 1, "y": 1, "no": 0, "false": 0, "0": 0, "n": 0})
        .fillna(0).astype(int)
    )

    svc_df["zip_codes"] = svc_df["zip_codes"].apply(_split_zipcodes)
    exploded = svc_df.explode("zip_codes").rename(columns={"zip_codes": "zip_code"})

    # No listed zips AND partial_county == 0 (covers full county) -> keep one row with '00000'
    no_zip_full = svc_df[(svc_df["zip_codes"].map(len) == 0) & (svc_df["partial_county"] == 0)][
        ["service_area_id", "state", "partial_county"]
    ].copy()
    no_zip_full["zip_code"] = "00000"

    svc_expanded = pd.concat(
        [
            exploded[["service_area_id", "state", "partial_county", "zip_code"]].dropna(subset=["zip_code"]),
            no_zip_full,
        ],
        ignore_index=True,
    ).drop_duplicates()

    # -------------------------
    # Merge Plan + Rate (by plan_id_root), then + Service Area
    # -------------------------
    merged = plan_df.merge(ref_rate, on="plan_id_root", how="left") \
                    .merge(svc_expanded, on=["service_area_id", "state"], how="left")

    # -------------------------
    # Final cleanup / select
    # -------------------------
    merged["premium_annual"] = pd.to_numeric(merged["premium_annual"], errors="coerce").fillna(0.0)
    merged["network_type"] = merged["network_type"].fillna("Unknown")
    merged["metal_tier"] = merged["metal_tier"].fillna("Unknown")
    merged["zip_code"] = merged["zip_code"].fillna("00000")

    merged = merged[[
        "plan_id", "plan_name", "metal_tier", "state", "zip_code",
        "premium_annual", "hsa_eligible", "network_type",
        "dental_only", "market_coverage", "partial_county", "sbc_url", "plan_brochure"
    ]].drop_duplicates()

    merged.to_parquet(OUT_PATH, index=False)
    print(f"✅ Saved merged CMS dataset → {OUT_PATH} (rows: {len(merged):,})")


if __name__ == "__main__":
    preprocess()
