"""Native (ante-hoc) explainability for the Explainable Boosting Machine.

Unlike the other three model families, EBM does not need a post-hoc SHAP
approximation -- it is a glass-box generalized additive model, so its own
term-importance and shape-function values ARE the explanation, exactly as
the model computed it. This module produces the same kind of figures SHAP
produces (global bar chart, per-feature effect plots) directly from
`ExplainableBoostingClassifier.explain_global()`, plus a helper to measure
rank agreement between EBM's native importances and another model's SHAP
mean-|value| importances (RQ2: do post-hoc and ante-hoc explanations agree?).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


def explain_ebm_model(
    pipeline: Pipeline,
    model_name: str,
    figures_dir: Path,
    n_shape_plots: int = 4,
) -> pd.DataFrame:
    """Returns a DataFrame of EBM term importance per feature (global,
    native explanation -- analogous in shape to the SHAP mean_abs_shap table
    returned by dac.explainability.shap_explain.explain_model).
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ebm = pipeline.named_steps["clf"]
    global_exp = ebm.explain_global()
    data = global_exp.data()
    names, scores = list(data["names"]), list(data["scores"])

    importance = pd.Series(scores, index=names).sort_values(ascending=False)
    importance.to_frame("ebm_importance").to_csv(figures_dir / f"{model_name}_ebm_importance.csv")

    # Global importance bar chart, styled like the SHAP bar chart for visual
    # comparability in the report.
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_xlabel("EBM term importance (mean absolute score)")
    ax.set_title(f"{model_name}: native EBM global term importance")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{model_name}_ebm_importance_bar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Shape-function plots for the top few single-feature (non-interaction)
    # terms -- the direct visualization of what the model learned per
    # feature, which is what makes EBM "glass-box".
    single_terms = [n for n in importance.index if " x " not in n and " & " not in n]
    plotted = 0
    for term_name in single_terms:
        if plotted >= n_shape_plots:
            break
        term_idx = names.index(term_name)
        term_data = global_exp.data(term_idx)
        bin_names = term_data.get("names")
        bin_scores = term_data.get("scores")
        if bin_names is None or bin_scores is None or len(bin_scores) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        if len(bin_names) == len(bin_scores) + 1:
            # Continuous feature: bin_names are edges -> step plot.
            x = np.asarray(bin_names[:-1], dtype=float)
            ax.step(x, bin_scores, where="post", color="#55A868")
        else:
            ax.bar([str(b) for b in bin_names], bin_scores, color="#55A868")
            plt.xticks(rotation=45, ha="right")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_title(f"{model_name}: shape function -- {term_name}")
        ax.set_ylabel("contribution to log-odds")
        fig.tight_layout()
        safe_name = term_name.replace("/", "_").replace(" ", "_")
        fig.savefig(figures_dir / f"{model_name}_ebm_shape_{safe_name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        plotted += 1

    return importance.to_frame("ebm_importance")


def compute_shap_ebm_agreement(shap_importance: pd.Series, ebm_importance: pd.Series) -> dict:
    """Spearman rank correlation between two feature-importance rankings
    over the features they have in common, feeding RQ2 (post-hoc SHAP vs.
    ante-hoc EBM agreement).
    """
    common = sorted(set(shap_importance.index) & set(ebm_importance.index))
    if len(common) < 3:
        logger.warning("Only %d common features between SHAP and EBM importances; agreement not meaningful.", len(common))
        return {"n_common_features": len(common), "spearman_r": None, "p_value": None}

    rho, p_value = spearmanr(shap_importance.loc[common], ebm_importance.loc[common])
    return {"n_common_features": len(common), "spearman_r": float(rho), "p_value": float(p_value)}
