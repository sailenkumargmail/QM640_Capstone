"""Builds a static flowchart showing how RQ1-RQ4 relate to each other and to
the overall pipeline/solution flow, for the Interim Report. Pure matplotlib
(no Graphviz binary dependency).

Usage:
    python scripts/build_rq_flowchart.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
OUT_PATH = REPO / "reports" / "figures" / "report_evidence" / "rq_solution_flowchart.png"

PIPELINE_COLOR = "#4C72B0"
MODEL_COLOR = "#DD8452"
RQ_COLOR = "#55A868"
OUTPUT_COLOR = "#C44E52"
TEXT_COLOR = "#1a1a1a"


def box(ax, xy, w, h, text, color, fontsize=9.5, fontweight="normal"):
    x, y = xy
    fb = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.3, edgecolor=color, facecolor=color, alpha=0.18,
    )
    ax.add_patch(fb)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight, color=TEXT_COLOR, wrap=True,
    )
    return (x, y, w, h)


def arrow(ax, start, end, color="#555555", style="-|>", lw=1.4, connectionstyle="arc3,rad=0.0", ls="-"):
    a = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14, linewidth=lw,
        color=color, connectionstyle=connectionstyle, linestyle=ls, zorder=1,
    )
    ax.add_patch(a)


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def build():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # --- Pipeline spine (left column) --------------------------------
    data_b = box(ax, (0.3, 7.2), 2.6, 0.8, "UCI Credit-Card\nDefault Data\n(30,000 rows)", PIPELINE_COLOR, fontweight="bold")
    eda_b = box(ax, (0.3, 5.9), 2.6, 0.8, "EDA\n(distributions, missingness,\nprotected-attribute gaps)", PIPELINE_COLOR)
    fe_b = box(ax, (0.3, 4.6), 2.6, 0.8, "Feature Engineering\n(utilization / repayment\nratios, AGE_GROUP)", PIPELINE_COLOR)
    models_b = box(
        ax, (0.3, 2.9), 2.6, 1.3,
        "Train + Tune 4 Models:\nLogistic Regression\nXGBoost | CatBoost | EBM",
        MODEL_COLOR, fontweight="bold",
    )
    reco_b = box(ax, (0.3, 0.3), 2.6, 1.1, "Synthesis:\nAccuracy-Fairness\nDecision Framework", OUTPUT_COLOR, fontweight="bold")

    arrow(ax, bottom(data_b), top(eda_b))
    arrow(ax, bottom(eda_b), top(fe_b))
    arrow(ax, bottom(fe_b), top(models_b))

    # --- RQ boxes (right side) ----------------------------------------
    rq1_b = box(
        ax, (4.4, 6.5), 3.5, 1.5,
        "RQ1 -- Model Comparison\nDo XGBoost / CatBoost / EBM\noutperform tuned Logistic\nRegression (ROC-AUC)?\nSelects the best model.",
        RQ_COLOR, fontweight="bold",
    )
    rq2_b = box(
        ax, (4.4, 4.6), 3.5, 1.5,
        "RQ2 -- Explainability\nDo SHAP (post-hoc) and EBM's\nnative term importances\n(ante-hoc) agree on the top\ndrivers of default risk?",
        RQ_COLOR, fontweight="bold",
    )
    rq3_b = box(
        ax, (4.4, 2.7), 3.5, 1.5,
        "RQ3 -- Fairness Audit\n+ Mitigation (best model)\nDoes the RQ1 best model violate\nthe four-fifths rule on SEX /\nAGE_GROUP, and does reweighing fix it?",
        RQ_COLOR, fontweight="bold",
    )
    rq4_b = box(
        ax, (4.4, 0.8), 3.5, 1.5,
        "RQ4 -- Cross-Model\nReplication\nDoes the identical reweighing\nrecipe from RQ3 improve parity\ncomparably across all 4 models?",
        RQ_COLOR, fontweight="bold",
    )

    arrow(ax, right(models_b), left(rq1_b), color=MODEL_COLOR, lw=1.6)
    arrow(ax, bottom(rq1_b), top(rq2_b), color=RQ_COLOR)
    arrow(ax, bottom(rq2_b), top(rq3_b), color=RQ_COLOR)
    arrow(ax, bottom(rq3_b), top(rq4_b), color=RQ_COLOR)

    # RQ1's selected best model feeds directly into RQ3's audit target too
    arrow(
        ax, (right(rq1_b)[0] + 0.05, right(rq1_b)[1]), (right(rq3_b)[0] + 0.05, right(rq3_b)[1]),
        color="#888888", lw=1.1, ls="--",
        connectionstyle="arc3,rad=0.55",
    )

    # --- Outputs feed back to the spine's synthesis box ----------------
    arrow(ax, left(rq4_b), right(reco_b), color=OUTPUT_COLOR, lw=1.6, connectionstyle="arc3,rad=-0.15")

    # --- Right-hand notes column (what each RQ outputs) -----------------
    box(ax, (8.6, 6.9) , 4.1, 1.1, "Output: best_model selection\n(Table 9, Preliminary Model Performance)", "#9b9b9b", fontsize=8.5)
    box(ax, (8.6, 5.0), 4.1, 1.1, "Output: Spearman rank agreement\n(SHAP vs. EBM top features)", "#9b9b9b", fontsize=8.5)
    box(ax, (8.6, 3.1), 4.1, 1.1, "Output: pre/post demographic parity\nratio + ROC-AUC cost (best model)", "#9b9b9b", fontsize=8.5)
    box(ax, (8.6, 1.2), 4.1, 1.1, "Output: pre/post parity for\nall 4 models (Table 10)", "#9b9b9b", fontsize=8.5)

    arrow(ax, right(rq1_b), left((8.6, 6.9, 4.1, 1.1)), color="#9b9b9b", lw=1.0)
    arrow(ax, right(rq2_b), left((8.6, 5.0, 4.1, 1.1)), color="#9b9b9b", lw=1.0)
    arrow(ax, right(rq3_b), left((8.6, 3.1, 4.1, 1.1)), color="#9b9b9b", lw=1.0)
    arrow(ax, right(rq4_b), left((8.6, 1.2, 4.1, 1.1)), color="#9b9b9b", lw=1.0)

    ax.set_title(
        "Solution Flow and RQ1-RQ4 Lineage: UCI Credit-Card Default Pipeline",
        fontsize=13, fontweight="bold", pad=14,
    )

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", OUT_PATH)


if __name__ == "__main__":
    build()
