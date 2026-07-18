"""End-to-end capstone pipeline orchestrator.

Runs, in order:
  1. Load Home Credit data (real if downloaded, else synthetic fallback)
  2. EDA -> reports/figures/home_credit/, reports/model_performance/eda_home_credit.md
  3. Feature engineering + train/val/test split
  4. Baseline training: Logistic Regression, XGBoost
  5. Hyperparameter tuning: RandomizedSearchCV (LR), Optuna (XGBoost)
  6. Evaluation: ROC-AUC, PR-AUC, F1, KS, Brier + diagnostic plots
  7. SHAP explainability on the best-performing tuned model
  8. Fairness audit (pre-mitigation) on CODE_GENDER / AGE_GROUP
  9. Bias mitigation via reweighing + re-train + re-audit (before/after comparison)
 10. HMDA: load data, train a pricing (higher-cost-loan) model, fairness-audit
    only (see dac.fairness.hmda_audit module docstring for why this is NOT a
    second "default" model, and why the real 2008-2017 extract's target is a
    pricing proxy rather than approval/denial)
 11. Persist trained models + a consolidated run_summary.json

Usage:
    python scripts/run_pipeline.py [--quick]

--quick shrinks tuning trial counts and synthetic row counts for a fast
smoke-test run (used by CI / test_pipeline.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dac.config import CONFIG
from dac.data.loader import load_home_credit, load_hmda
from dac.eda.eda_report import run_eda
from dac.explainability.shap_explain import explain_model
from dac.fairness.audit import audit_fairness
from dac.fairness.hmda_audit import run_hmda_fairness_audit
from dac.fairness.mitigation import compute_reweighing_weights
from dac.features.engineering import build_preprocessor, engineer_home_credit_features, split_feature_columns
from dac.models.evaluate import compare_models, compute_metrics, plot_evaluation_suite
from dac.models.train import build_logistic_regression_pipeline, build_xgboost_pipeline, compute_scale_pos_weight, fit
from dac.models.tune import tune_logistic_regression, tune_xgboost
from dac.utils.logging_utils import get_logger

logger = get_logger("run_pipeline")


def main(quick: bool = False) -> dict:
    t0 = time.time()
    seed = CONFIG["seed"]
    figures_dir = CONFIG["paths"]["figures_dir"]
    metrics_dir = CONFIG["paths"]["metrics_dir"]
    models_dir = CONFIG["paths"]["models_dir"]

    n_trials = 5 if quick else CONFIG["tuning"]["n_trials"]
    cv_folds = 3 if quick else CONFIG["tuning"]["cv_folds"]

    summary: dict = {}

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 1: Load Home Credit data\n" + "=" * 70)
    hc_cfg = CONFIG["data"]["home_credit"]
    df, is_synthetic = load_home_credit()
    summary["home_credit_is_synthetic"] = is_synthetic
    summary["home_credit_shape"] = list(df.shape)

    # ------------------------------------------------------------------
    # 2. EDA
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 2: EDA\n" + "=" * 70)
    eda_stats = run_eda(
        df,
        target_col=hc_cfg["target_col"],
        protected_attributes=hc_cfg["protected_attributes"],
        figures_dir=figures_dir,
        report_path=metrics_dir / "eda_home_credit.md",
        dataset_name="home_credit",
    )
    summary["home_credit_eda"] = eda_stats

    # ------------------------------------------------------------------
    # 3. Feature engineering + split
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 3: Feature engineering + split\n" + "=" * 70)
    df_fe = engineer_home_credit_features(df)

    target_col = hc_cfg["target_col"]
    id_col = hc_cfg["id_col"]
    protected_attributes = hc_cfg["protected_attributes"]
    exclude_cols = [id_col, *protected_attributes]
    numeric_cols, categorical_cols = split_feature_columns(df_fe, target_col, exclude_cols)
    logger.info("Home Credit model features: %d numeric, %d categorical", len(numeric_cols), len(categorical_cols))

    X = df_fe[numeric_cols + categorical_cols]
    y = df_fe[target_col]
    sensitive = df_fe[protected_attributes]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=CONFIG["split"]["test_size"], random_state=seed, stratify=y
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # ------------------------------------------------------------------
    # 4. Baseline training
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 4: Baseline training\n" + "=" * 70)
    lr_pipeline = build_logistic_regression_pipeline(preprocessor, max_iter=CONFIG["models"]["logistic_regression"]["max_iter"])
    lr_baseline = fit(lr_pipeline, X_train, y_train, "logistic_regression_baseline")

    xgb_pipeline = build_xgboost_pipeline(
        preprocessor,
        n_estimators=CONFIG["models"]["xgboost"]["n_estimators"],
        scale_pos_weight=compute_scale_pos_weight(y_train),
    )
    xgb_baseline = fit(xgb_pipeline, X_train, y_train, "xgboost_baseline")

    baseline_results = {}
    for trained in (lr_baseline, xgb_baseline):
        proba = trained.pipeline.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test.to_numpy(), proba)
        baseline_results[trained.name] = m
        plot_evaluation_suite(y_test.to_numpy(), proba, trained.name, figures_dir / "home_credit")
    compare_models(baseline_results, metrics_dir / "baseline")
    summary["baseline_metrics"] = baseline_results

    # ------------------------------------------------------------------
    # 5. Hyperparameter tuning
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 5: Hyperparameter tuning\n" + "=" * 70)
    lr_tuned = tune_logistic_regression(
        preprocessor, X_train, y_train, n_iter=15 if not quick else 4, cv_folds=cv_folds, seed=seed
    )
    xgb_tuned = tune_xgboost(preprocessor, X_train, y_train, n_trials=n_trials, cv_folds=cv_folds, seed=seed)

    lr_tuned.pipeline.fit(X_train, y_train)
    xgb_tuned.pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Evaluation of tuned models
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 6: Evaluate tuned models\n" + "=" * 70)
    tuned_pipelines = {"logistic_regression_tuned": lr_tuned.pipeline, "xgboost_tuned": xgb_tuned.pipeline}
    tuned_results = {}
    for name, pipeline in tuned_pipelines.items():
        proba = pipeline.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test.to_numpy(), proba)
        tuned_results[name] = m
        plot_evaluation_suite(y_test.to_numpy(), proba, name, figures_dir / "home_credit")
    all_results = {**baseline_results, **tuned_results}
    compare_models(all_results, metrics_dir)
    summary["tuned_metrics"] = tuned_results
    summary["tuning_best_params"] = {"logistic_regression": lr_tuned.best_params, "xgboost": xgb_tuned.best_params}

    best_model_name = max(tuned_results, key=lambda k: tuned_results[k]["roc_auc"])
    best_pipeline = tuned_pipelines[best_model_name]
    logger.info("Best model by ROC-AUC: %s", best_model_name)
    summary["best_model"] = best_model_name

    joblib.dump(lr_tuned.pipeline, models_dir / "logistic_regression_tuned.joblib")
    joblib.dump(xgb_tuned.pipeline, models_dir / "xgboost_tuned.joblib")

    # ------------------------------------------------------------------
    # 7. SHAP explainability (best model)
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 7: SHAP explainability\n" + "=" * 70)
    shap_importance = explain_model(
        best_pipeline, X_train, X_test, best_model_name, figures_dir / "home_credit" / "shap"
    )
    summary["shap_top_features"] = shap_importance.head(10)["mean_abs_shap"].to_dict()

    # ------------------------------------------------------------------
    # 8. Fairness audit (pre-mitigation)
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 8: Fairness audit (pre-mitigation)\n" + "=" * 70)
    y_pred_best = (best_pipeline.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    favorable_label = CONFIG["fairness"]["favorable_label"]
    pre_mitigation_fairness = {}
    for attr in protected_attributes:
        pre_mitigation_fairness[attr] = audit_fairness(
            y_true=y_test.to_numpy(),
            y_pred=y_pred_best,
            sensitive_features=sens_test[attr],
            model_name=f"{best_model_name}_pre_mitigation",
            attribute_name=attr,
            figures_dir=figures_dir / "home_credit" / "fairness",
            metrics_dir=metrics_dir / "fairness",
            favorable_label=favorable_label,
        )
    summary["fairness_pre_mitigation"] = pre_mitigation_fairness

    # ------------------------------------------------------------------
    # 9. Bias mitigation (reweighing) + re-audit
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 9: Bias mitigation (reweighing)\n" + "=" * 70)
    mitigation_attr = protected_attributes[0]
    sample_weights = compute_reweighing_weights(y_train, sens_train[mitigation_attr])

    if "xgboost" in best_model_name:
        mitigated_pipeline = build_xgboost_pipeline(
            preprocessor, n_estimators=xgb_tuned.best_params.get("n_estimators", 400),
            scale_pos_weight=compute_scale_pos_weight(y_train),
        )
    else:
        mitigated_pipeline = build_logistic_regression_pipeline(preprocessor)

    mitigated_pipeline.fit(X_train, y_train, clf__sample_weight=sample_weights)
    joblib.dump(mitigated_pipeline, models_dir / f"{best_model_name}_mitigated.joblib")

    proba_mitigated = mitigated_pipeline.predict_proba(X_test)[:, 1]
    y_pred_mitigated = (proba_mitigated >= 0.5).astype(int)
    mitigated_perf = compute_metrics(y_test.to_numpy(), proba_mitigated)
    plot_evaluation_suite(y_test.to_numpy(), proba_mitigated, f"{best_model_name}_mitigated", figures_dir / "home_credit")
    summary["mitigated_performance"] = mitigated_perf

    post_mitigation_fairness = {}
    for attr in protected_attributes:
        post_mitigation_fairness[attr] = audit_fairness(
            y_true=y_test.to_numpy(),
            y_pred=y_pred_mitigated,
            sensitive_features=sens_test[attr],
            model_name=f"{best_model_name}_post_mitigation",
            attribute_name=attr,
            figures_dir=figures_dir / "home_credit" / "fairness",
            metrics_dir=metrics_dir / "fairness",
            favorable_label=favorable_label,
        )
    summary["fairness_post_mitigation"] = post_mitigation_fairness
    summary["fairness_mitigation_attribute"] = mitigation_attr

    dp_before = pre_mitigation_fairness[mitigation_attr]["demographic_parity_ratio"]
    dp_after = post_mitigation_fairness[mitigation_attr]["demographic_parity_ratio"]
    logger.info(
        "Reweighing effect on %s demographic parity ratio: %.3f -> %.3f (target >= %.2f); "
        "ROC-AUC: %.4f -> %.4f",
        mitigation_attr, dp_before, dp_after, CONFIG["fairness"]["disparate_impact_threshold"],
        tuned_results[best_model_name]["roc_auc"], mitigated_perf["roc_auc"],
    )

    # ------------------------------------------------------------------
    # 10. HMDA fairness-only audit
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 10: HMDA fairness-only audit\n" + "=" * 70)
    hmda_cfg = CONFIG["data"]["hmda"]
    hmda_df, hmda_is_synthetic = load_hmda()
    summary["hmda_is_synthetic"] = hmda_is_synthetic
    summary["hmda_shape"] = list(hmda_df.shape)

    hmda_eda_stats = run_eda(
        hmda_df,
        target_col=hmda_cfg["target_col"],
        protected_attributes=hmda_cfg["protected_attributes"],
        figures_dir=figures_dir,
        report_path=metrics_dir / "eda_hmda.md",
        dataset_name="hmda",
    )
    summary["hmda_eda"] = hmda_eda_stats

    hmda_results = run_hmda_fairness_audit(
        hmda_df,
        target_col=hmda_cfg["target_col"],
        id_col=hmda_cfg["id_col"],
        protected_attributes=hmda_cfg["protected_attributes"],
        figures_dir=figures_dir / "hmda",
        metrics_dir=metrics_dir / "fairness",
        seed=seed,
        favorable_label=CONFIG["fairness"]["hmda_favorable_label"],
    )
    summary["hmda_results"] = hmda_results

    # ------------------------------------------------------------------
    # Wrap up
    # ------------------------------------------------------------------
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    (metrics_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Pipeline complete in %.1fs. Summary: %s", summary["elapsed_seconds"], metrics_dir / "run_summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test run with fewer tuning trials")
    args = parser.parse_args()
    main(quick=args.quick)
