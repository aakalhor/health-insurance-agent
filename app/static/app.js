async function recommend() {
  const btn = document.getElementById("btnRecommend");
  const toast = (msg) => {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.style.display = "block";
    setTimeout(() => (el.style.display = "none"), 2200);
  };

  const age = parseInt(document.getElementById("age").value || "0", 10);
  const sex = document.getElementById("sex").value || "female";
  const zip = document.getElementById("zip").value || "";
  const smoker = document.getElementById("smoker").value === "true";
  const bmi = parseFloat(document.getElementById("bmi").value || "0");
  const children = parseInt(document.getElementById("children").value || "0", 10);

  const metalSelect = document.getElementById("metalPref");
  const metalPrefs = Array.from(metalSelect.selectedOptions).map((o) => o.value);

  if (!zip) {
    toast("Enter a ZIP code");
    return;
  }

  const body = { age, sex, zip_code: zip, smoker, bmi, children };
  const params = new URLSearchParams();
  params.set("top_k", "5");
  metalPrefs.forEach((m) => params.append("metal_tier_preference", m));

  btn.disabled = true;
  try {
    const resp = await fetch(`/recommend?${params.toString()}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const t = await resp.text();
      throw new Error(`Recommend failed: ${resp.status} ${t}`);
    }
    const data = await resp.json();
    renderBand(data.predictions);
    renderExplanation(data.explanation);
    renderPlans(data.top_plans);
  } catch (e) {
    console.error(e);
    toast(e.message || "Failed to recommend");
  } finally {
    btn.disabled = false;
  }
}

function renderBand(preds) {
  if (!preds) return;
  const band = document.getElementById("band");
  band.textContent = `Range: $${fmt(preds.p25)}–$${fmt(preds.p75)} (median $${fmt(preds.p50)})`;
}

function renderExplanation(text) {
  const el = document.getElementById("explanation");
  el.style.display = "block";
  el.textContent = text; // already simple Markdown-like; we just show it as pre-wrapped text
}

function renderPlans(plans) {
  const root = document.getElementById("plans");
  const empty = document.getElementById("empty");
  root.innerHTML = "";
  if (!plans || plans.length === 0) {
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";
  for (const p of plans) {
    const card = document.createElement("div");
    card.className = "plan";
    const premium = fmt(p.premium_annual);
    const total = fmt(p.total_cost_estimate);
    card.innerHTML = `
      <h4>${escapeHtml(p.plan_name)} <span class="badge">${escapeHtml(p.metal_tier)}</span></h4>
      <div class="grid2">
        <div><small class="muted">Total est.</small><div>$${total}</div></div>
        <div><small class="muted">Premium</small><div>$${premium}</div></div>
      </div>
      <div class="hr"></div>
      <div><small class="muted">${escapeHtml(p.notes || "")}</small></div>
      <div style="margin-top:10px;">
        <button data-plan-id="${p.plan_id}">View details</button>
      </div>
    `;
    card.querySelector("button").addEventListener("click", () => viewPlan(p.plan_id));
    root.appendChild(card);
  }
}

async function viewPlan(planId) {
  const toast = (msg) => {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.style.display = "block";
    setTimeout(() => (el.style.display = "none"), 2200);
  };
  try {
    const resp = await fetch(`/plan/${encodeURIComponent(planId)}`);
    if (!resp.ok) {
      throw new Error(`Plan not found: ${planId}`);
    }
    const d = await resp.json();
    const lines = [];
    lines.push(`${d.plan_name} (${d.metal_tier}) — ${d.network_type}`);
    lines.push(`Annual premium: $${fmt(d.premium_annual)}`);
    if (d.hsa_eligible) lines.push("HSA-eligible plan");
    if (d.sbc_url) lines.push(`SBC: ${d.sbc_url}`);
    if (d.plan_brochure) lines.push(`Brochure: ${d.plan_brochure}`);
    alert(lines.join("\n"));
  } catch (e) {
    console.error(e);
    toast(e.message || "Failed to load plan");
  }
}

// helpers
function fmt(x) {
  if (x == null || isNaN(x)) return "0";
  return Number(x).toLocaleString(undefined, { maximumFractionDigits: 0 });
}
function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

document.getElementById("btnRecommend").addEventListener("click", recommend);
