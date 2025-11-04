# training/plot_training_report.py
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "data"
MODEL_DIR = DATA_DIR / "models"

CURVES = {
    "p25": MODEL_DIR / "training_curves_lgb_25.csv",
    "p50": MODEL_DIR / "training_curves_lgb_50.csv",
    "p75": MODEL_DIR / "training_curves_lgb_75.csv",
}
EVAL_JSON = MODEL_DIR / "metrics_eval.json"          # from eval_spend_model.py
TRAIN_JSON = MODEL_DIR / "metrics_summary.json"      # from train_spend_model.py (optional)
OUT_PNG = MODEL_DIR / "report_training_eval.png"


def _fmt_money(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)


def load_curves():
    out = {}
    for tag, path in CURVES.items():
        if path.exists():
            df = pd.read_csv(path)
            # Normalize expected columns
            if "iteration" not in df.columns:
                df["iteration"] = range(1, len(df) + 1)
            if "train_quantile" not in df.columns:
                df["train_quantile"] = pd.NA
            if "valid_quantile" not in df.columns:
                df["valid_quantile"] = pd.NA
            out[tag] = df
    return out


def load_json(path: Path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def main():
    curves = load_curves()
    eval_metrics = load_json(EVAL_JSON) or {}
    train_metrics = load_json(TRAIN_JSON) or {}

    # ---- Figure layout (3 rows):
    # Row 1: Learning curves (quantile loss) side-by-side for p25/p50/p75
    # Row 2: Bar chart of MAE (dollars) for test
    # Row 3: Summary text (R2, RMSE, pinball) + meta
    fig = plt.figure(figsize=(12, 12))

    # --- Row 1: Learning curves
    gs = fig.add_gridspec(3, 3, height_ratios=[2.1, 1.3, 1.6], hspace=0.35, wspace=0.25)

    for j, tag in enumerate(["p25", "p50", "p75"]):
        ax = fig.add_subplot(gs[0, j])
        df = curves.get(tag)
        ax.set_title(f"Learning Curve — {tag.upper()} (quantile loss)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (log space)")
        if df is not None:
            if "train_quantile" in df.columns and df["train_quantile"].notna().any():
                ax.plot(df["iteration"], df["train_quantile"], label="train")
            if "valid_quantile" in df.columns and df["valid_quantile"].notna().any():
                ax.plot(df["iteration"], df["valid_quantile"], label="valid")
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No curve file", ha="center", va="center", transform=ax.transAxes)

    # --- Row 2: MAE bars (test)
    ax_mae = fig.add_subplot(gs[1, :])
    tags = ["p25", "p50", "p75"]
    maes = []
    for tag in tags:
        m = (eval_metrics.get(tag) or {})
        maes.append(m.get("mae_dollars", float("nan")))
    ax_mae.bar(tags, maes)
    ax_mae.set_title("Test MAE (dollars)")
    ax_mae.set_ylabel("MAE ($)")
    for i, v in enumerate(maes):
        if pd.notna(v):
            ax_mae.text(i, v, _fmt_money(v), ha="center", va="bottom")

    # --- Row 3: Text summary
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")

    lines = []
    # Test metrics
    if eval_metrics:
        lines.append("**Test metrics**")
        for tag in tags:
            m = eval_metrics.get(tag, {})
            if m:
                lines.append(
                    f"• {tag.upper()}: MAE={_fmt_money(m.get('mae_dollars', float('nan')))}, "
                    f"RMSE={_fmt_money(m.get('rmse_dollars', float('nan')))}, "
                    f"R²={m.get('r2', float('nan')):.4f}, "
                    f"pinball(log)={m.get('pinball_loss_log', float('nan')):.5f}"
                )
        cr = eval_metrics.get("quantile_crossing_rate")
        if cr is not None:
            lines.append(f"• Quantile crossing rate: {cr:.2%}")
        n_test = eval_metrics.get("n_test")
        if n_test is not None:
            lines.append(f"• N_test: {n_test}")
        lines.append("")

    # Validation metrics
    if train_metrics:
        lines.append("**Validation metrics (from training)**")
        for tag in tags:
            m = train_metrics.get(tag, {})
            if m:
                lines.append(
                    f"• {tag.upper()}: MAE={_fmt_money(m.get('mae_dollars', float('nan')))}, "
                    f"RMSE={_fmt_money(m.get('rmse_dollars', float('nan')))}, "
                    f"R²={m.get('r2', float('nan')):.4f}, "
                    f"pinball(log)={m.get('pinball_loss_log', float('nan')):.5f}"
                )

    # Render text
    y = 0.95
    for ln in lines:
        ax_txt.text(0.02, y, ln, va="top", ha="left")
        y -= 0.07

    fig.suptitle("Spend Model — Training & Evaluation Report", y=0.99, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"✅ Saved report → {OUT_PNG}")


if __name__ == "__main__":
    main()
