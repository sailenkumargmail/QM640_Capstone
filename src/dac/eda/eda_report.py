"""Exploratory Data Analysis: figures + a markdown summary report.

Covers: shape/dtypes, missingness, target balance, numeric distributions,
categorical breakdowns, correlation with target, and a protected-attribute
default-rate breakdown (the EDA finding that motivates the fairness-audit
stage later in the pipeline).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_eda(
    df: pd.DataFrame,
    target_col: str,
    protected_attributes: list[str],
    figures_dir: Path,
    report_path: Path,
    dataset_name: str,
) -> dict:
    logger.info("Running EDA for %s (%d rows, %d cols)", dataset_name, *df.shape)
    figures_dir = Path(figures_dir) / dataset_name
    lines: list[str] = [f"# EDA Report: {dataset_name}\n"]

    # --- Shape / dtypes -------------------------------------------------
    lines.append(f"- Rows: **{df.shape[0]:,}**, Columns: **{df.shape[1]}**")
    n_dupes = df.duplicated().sum()
    lines.append(f"- Duplicate rows: **{n_dupes}**")

    # --- Target balance ---------------------------------------------------
    target_counts = df[target_col].value_counts(normalize=True).sort_index()
    lines.append(f"\n## Target balance (`{target_col}`)\n")
    lines.append(target_counts.to_frame("proportion").to_markdown())

    fig, ax = plt.subplots(figsize=(4, 4))
    df[target_col].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title(f"{dataset_name}: target distribution")
    ax.set_xlabel(target_col)
    _save(fig, figures_dir / "target_distribution.png")

    # --- Missingness --------------------------------------------------
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    lines.append("\n## Missingness (top 20 columns)\n")
    if len(missing):
        lines.append(missing.head(20).to_frame("missing_fraction").to_markdown())
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * min(20, len(missing)))))
        missing.head(20).sort_values().plot(kind="barh", ax=ax, color="#DD8452")
        ax.set_title(f"{dataset_name}: missing value fraction (top 20)")
        _save(fig, figures_dir / "missingness.png")
    else:
        lines.append("No missing values.")

    # --- Numeric distributions -----------------------------------------
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col][:12]
    if numeric_cols:
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col].dropna(), ax=axes[i], kde=True, color="#4C72B0")
            axes[i].set_title(col, fontsize=9)
        for j in range(len(numeric_cols), len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{dataset_name}: numeric feature distributions")
        fig.tight_layout()
        _save(fig, figures_dir / "numeric_distributions.png")

    # --- Correlation with target -----------------------------------------
    numeric_df = df.select_dtypes(include="number")
    if target_col in numeric_df.columns and numeric_df.shape[1] > 1:
        corr_with_target = (
            numeric_df.corr(numeric_only=True)[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        )
        lines.append("\n## Top correlations with target\n")
        lines.append(corr_with_target.head(15).to_frame("corr").to_markdown())

        fig, ax = plt.subplots(figsize=(10, 8))
        top_corr_cols = corr_with_target.head(15).index.tolist() + [target_col]
        sns.heatmap(numeric_df[top_corr_cols].corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title(f"{dataset_name}: correlation heatmap (top 15 features + target)")
        _save(fig, figures_dir / "correlation_heatmap.png")

    # --- Protected attribute breakdown -----------------------------------
    lines.append("\n## Target rate by protected attribute\n")
    for attr in protected_attributes:
        if attr not in df.columns:
            continue
        rate_by_group = df.groupby(attr)[target_col].agg(["mean", "count"]).sort_values("mean", ascending=False)
        lines.append(f"\n### {attr}\n")
        lines.append(rate_by_group.to_markdown())

        fig, ax = plt.subplots(figsize=(6, 4))
        rate_by_group["mean"].plot(kind="bar", ax=ax, color="#55A868")
        ax.set_ylabel(f"mean({target_col})")
        ax.set_title(f"{dataset_name}: {target_col} rate by {attr}")
        _save(fig, figures_dir / f"target_rate_by_{attr}.png")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n\n".join(lines), encoding="utf-8")
    logger.info("EDA report written to %s", report_path)

    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "n_duplicates": int(n_dupes),
        "target_rate": float(df[target_col].mean()),
        "n_missing_cols": int(len(missing)),
    }
