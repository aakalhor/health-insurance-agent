# 🏥 Health Insurance Agent — Phase 3

**An intelligent, local-first health insurance assistant** that predicts medical spending, ranks available insurance plans from CMS 2025 Public Use Files (PUFs), and explains trade-offs in plain English.

Built with **FastAPI + LightGBM + SQLite + HTML/JS frontend**.

---

## 🚀 Features

✅ **Predict healthcare spend**  
Quantile regression model (LightGBM) estimates the 25th, 50th, and 75th percentile of expected yearly spending (`p25`, `p50`, `p75`) for a user profile.

✅ **Rank CMS 2025 plans**  
Filters, merges, and ranks ACA marketplace plans from the latest CMS Plan, Rate, and Service Area PUFs using ZIP-specific coverage.

✅ **Explain trade-offs (LLM explainer)**  
A local lightweight reasoning module explains plan pros/cons and user fit — no API calls, no external LLM required.

✅ **Frontend demo**  
A clean static HTML + JS dashboard for submitting profiles, viewing explanations, and exploring top-ranked plans with plan detail pop-ups.

✅ **Local, privacy-first**  
All data and models run locally on your machine; no cloud dependencies.

---

## 🧱 Architecture Overview

```
FastAPI backend
│
├── /data_pipeline
│   ├── prep_cms_pufs_2025.py → Cleans & merges CMS Plan, Rate, and Service Area PUFs
│   └── build_plans_sqlite.py → Converts Parquet → SQLite
│
├── /training
│   └── train_spend_model.py → Trains quantile LightGBM models (p25, p50, p75)
│
├── /app
│   ├── main.py → FastAPI routes (predict, rank, explain, plan detail)
│   ├── /services
│   │   ├── spend_model.py → Loads trained LightGBM quantile models
│   │   ├── ranker.py → Filters and ranks plans by total cost
│   │   ├── llm_explainer.py → Generates natural-language explanations
│   │   └── plan_lookup.py → Fetches plan details (SBC URLs, brochures, etc.)
│   ├── /data → Models + SQLite database + cleaned CMS data
│   ├── /static → Frontend (index.html + app.js)
│   └── schemas.py → Pydantic models for validation & responses
│
└── /app/static
    ├── index.html → Frontend UI
    └── app.js → JS logic to call API & render results
```

---

## 🧩 Tech Stack

| Layer | Technology | Purpose |
|-------|-------------|----------|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) | REST API and routing |
| **ML Model** | [LightGBM](https://lightgbm.readthedocs.io/) | Quantile regression models |
| **Data Storage** | [SQLite](https://www.sqlite.org/) | Local plan database |
| **Data Prep** | [Pandas](https://pandas.pydata.org/) | CMS dataset cleanup |
| **Frontend** | HTML + Vanilla JS | Lightweight interactive dashboard |
| **Infra** | Local (no cloud) | Runs fully offline once models/data are built |

---

## ⚙️ Installation

### 1. Clone repo
```bash
git clone https://github.com/<your-username>/health-insurance-agent.git
cd health-insurance-agent
```

### 2. Create environment
```bash
python -m venv .venv
.\.venv\Scriptsctivate    # Windows
# or
source .venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare CMS PUF data

Place the following 2025 CMS CSVs in `app/data/`:
```
Plan_Attributes_PUF.csv
Rate_PUF.csv
Service_Area_PUF.csv
```

### 5. Run data pipeline
```bash
python data_pipeline/prep_cms_pufs_2025.py
python data_pipeline/build_plans_sqlite.py
```

### 6. Train the model (optional — already included in repo)
```bash
python training/train_spend_model.py
```
Models are saved under `app/data/models/lgb_p25.txt`, `lgb_p50.txt`, `lgb_p75.txt`.

---

## ▶️ Running the App

Start FastAPI server:
```bash
uvicorn app.main:app --reload
```

Then open:  
**http://127.0.0.1:8000/**

---

## 🖥️ Frontend Demo

**Features:**
- Input form for age, sex, zip, smoker, BMI, children  
- “Recommend & Explain” button calls `/recommend`
- Displays:
  - Predicted spending band (p25–p75)
  - Explanation text
  - Top plan cards (total + premium)
  - “View details” button fetches `/plan/{plan_id}` with SBC URL + brochure links

---

## 🧠 API Endpoints

| Method | Endpoint | Description |
|--------|-----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict quantile spending for a user profile |
| POST | `/rank` | Rank top 5 plans by total estimated cost |
| POST | `/explain` | Return textual explanation only |
| POST | `/recommend` | Predict + rank + explain combined |
| GET | `/plan/{plan_id}` | Get detailed info for one plan (SBC, brochure, flags) |

---

## 🧮 Example Request

```bash
curl -X POST "http://127.0.0.1:8000/recommend" ^
     -H "Content-Type: application/json" ^
     -d "{ \"age\": 40, \"sex\": \"female\", \"zip_code\": \"49457\", \"smoker\": false, \"bmi\": 26, \"children\": 1 }"
```

### Example Response
```json
{
  "predictions": { "p25": 5880.58, "p50": 5840.93, "p75": 5802.09 },
  "top_plans": [
    {
      "plan_id": "40047MI0010001-00",
      "plan_name": "Gold 1",
      "metal_tier": "Gold",
      "premium_annual": 4525.74,
      "total_cost_estimate": 10366.67,
      "notes": "Network: HMO"
    }
  ],
  "explanation": "Top plan options for your profile..."
}
```

---

## 🧩 How It Works

### Feature Engineering
Encodes age, BMI, smoker status, region dummies, etc.

### Spend Prediction
Quantile LightGBM models estimate p25/p50/p75 annual spending.

### Plan Ranking
Combines predicted medical cost + annual premium → total expected cost.  
Filters out dental-only and non-individual plans.

### Explanation Generation
Local deterministic rules describe metals, networks, and fit rationale.

### Frontend
Calls `/recommend`, renders explanation & plans interactively.

---

## 📁 Key Data Fields

| Field | Description |
|--------|-------------|
| metal_tier | Bronze, Silver, Gold, Platinum, Catastrophic |
| network_type | HMO, PPO, EPO, POS |
| hsa_eligible | 1 if plan supports Health Savings Account |
| dental_only | 1 if dental-only plan |
| market_coverage | Individual / SHOP |
| sbc_url | Summary of Benefits and Coverage (official PDF) |
| plan_brochure | Plan marketing brochure |

---

## 🧠 Future Enhancements

✅ Add deductible/copay parsing from SBCs (Phase 4)  
✅ Add authentication + user profile saving  
✅ Add API caching & performance optimization  
✅ Containerize with Docker  
✅ Integrate lightweight local LLM (Gemma 2 or Phi-3-mini) for richer explanations  

---

## 🧾 License

MIT License © 2025 — Developed as an academic/engineering project for AI-powered Health Insurance Plan Recommendation.

---

## 👩‍💻 Maintainer

**Amirali Kalhor**  
M.S. Computer Science — CSULB  
[GitHub](https://github.com/aakalhor) · [LinkedIn](https://www.linkedin.com/in/amirali-kalhor)
