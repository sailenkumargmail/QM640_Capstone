"""SHAP-based explainability for the trained pipelines.

Produces global explanations (summary/beeswarm, bar of mean |SHAP|) and a
handful of local (single-prediction) waterfall explanations, saved as
figures for the capstone report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _transform(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocessor = pipeline.named_steps["preprocess"]
    X_t = preprocessor.transform(X)
    if isinstance(X_t, pd.DataFrame):
        X_t = X_t.to_numpy()
    feature_names = list(preprocessor.get_feature_names_out())
    return X_t, feature_names


def explain_model(
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    model_name: str,
    figures_dir: Path,
    max_background: int = 200,
    max_explain: int = 500,
    n_local_examples: int = 3,
) -> pd.DataFrame:
    """Returns a DataFrame of mean |SHAP value| per feature (global importance)."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clf = pipeline.named_steps["clf"]
    X_bg_t, feature_names = _transform(pipeline, X_background.sample(
        n=min(max_background, len(X_background)), random_state=42
    ))
    X_ex_t, _ = _transform(pipeline, X_explain.sample(n=min(max_explain, len(X_explain)), random_state=42))

    logger.info("Computing SHAP values for %s (%d background, %d explain rows)", model_name, len(X_bg_t), len(X_ex_t))

    model_type = type(clf).__name__
    if "XGB" in model_type or "LGBM" in model_type or "CatBoost" in model_type:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_ex_t)
    else:
        explainer = shap.LinearExplainer(clf, X_bg_t)
        shap_values = explainer.shap_values(X_ex_t)

    if isinstance(shap_values, list):  # binary-classifier list output, take positive class
        shap_values = shap_values[1]

    # Global importance: bar chart
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_ex_t, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
    plt.title(f"{model_name}: global feature importance (mean |SHAP|)")
    plt.tight_layout()
    fig.savefig(figures_dir / f"{model_name}_shap_bar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Global importance: beeswarm (distribution + direction of effect)
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_ex_t, feature_names=feature_names, show=False, max_display=20)
    plt.title(f"{model_name}: SHAP summary (beeswarm)")
    plt.tight_layout()
    fig.savefig(figures_dir / f"{model_name}_shap_beeswarm.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Local explanations for a few individual predictions
    for i in range(min(n_local_examples, len(X_ex_t))):
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[-1]
        explanation = shap.Explanation(
            values=shap_values[i],
            base_values=expected_value,
            data=X_ex_t[i],
            feature_names=feature_names,
        )
        fig = plt.figure(figsize=(9, 5))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title(f"{model_name}: local explanation, example #{i}")
        plt.tight_layout()
        fig.savefig(figures_dir / f"{model_name}_shap_local_example_{i}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)
    mean_abs_shap.to_frame("mean_abs_shap").to_csv(figures_dir / f"{model_name}_shap_importance.csv")
    return mean_abs_shap.to_frame("mean_abs_shap")
