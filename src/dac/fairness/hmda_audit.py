"""HMDA fairness-only pipeline: train an approval/denial model on HMDA and
fairness-audit it. This deliberately does NOT call the target "default" --
HMDA's action_taken field records the origination decision, not loan
performance, so it cannot be conflated with Home Credit's TARGET (see
Synopsis_Credit_Default_XAI_Bias_Mitigation.docx, Background and Context,
and the mentor/reviewer critique's Cross-Question 1). This module's sole
purpose is testing whether the SAME fairness-auditing methodology used on
Home Credit generalizes to a second, independently-labeled dataset and
decision task.

Protected attributes (derived_race, derived_sex) are excluded from the model
features (used only for the post-hoc fairness audit), matching standard fair-
lending practice of not using protected class directly in underwriting
models.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from dac.features.engineering import build_preprocessor, engineer_hmda_features, split_feature_columns
from dac.models.evaluate import compute_metrics, plot_evaluation_suite
from dac.models.train import build_logistic_regression_pipeline, build_xgboost_pipeline, compute_scale_pos_weight, fit
from dac.fairness.audit import audit_fairness
from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_hmda_fairness_audit(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    protected_attributes: list[str],
    figures_dir: Path,
    metrics_dir: Path,
    seed: int = 42,
) -> dict:
    df = engineer_hmda_features(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    exclude_cols = [id_col, target_col, *protected_attributes]
    numeric_cols, categorical_cols = split_feature_columns(df, target_col, exclude_cols)
    logger.info("HMDA model features: %d numeric, %d categorical (protected attrs excluded)", len(numeric_cols), len(categorical_cols))

    X = df[numeric_cols + categorical_cols]
    y = df[target_col]
    sensitive = df[protected_attributes]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.20, random_state=seed, stratify=y
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = build_xgboost_pipeline(
        preprocessor, n_estimators=300, scale_pos_weight=compute_scale_pos_weight(y_train)
    )
    trained = fit(pipeline, X_train, y_train, name="hmda_xgboost")

    y_proba = trained.pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_proba)
    logger.info("HMDA approval-model metrics: %s", metrics)
    plot_evaluation_suite(y_test.to_numpy(), y_proba, "hmda_xgboost", figures_dir)

    y_pred = (y_proba >= 0.5).astype(int)
    fairness_results = {}
    for attr in protected_attributes:
        fairness_results[attr] = audit_fairness(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            sensitive_features=sens_test[attr],
            model_name="hmda_xgboost",
            attribute_name=attr,
            figures_dir=figures_dir,
            metrics_dir=metrics_dir,
            favorable_label=1,  # 1 = originated, the favorable HMDA outcome
        )

    return {"performance": metrics, "fairness": fairness_results}
