"""End-to-end capstone pipeline orchestrator (UCI credit-card default data).

Runs, in order:
  1. Load UCI "default of credit card clients" data (real if prepared, else
     synthetic fallback)
  2. EDA -> reports/figures/uci_credit/, reports/model_performance/eda_uci_credit.md
  3. Feature engineering + train/test split
  4. Baseline training: Logistic Regression, XGBoost, CatBoost, EBM
  5. Hyperparameter tuning: RandomizedSearchCV (LR), Optuna (XGBoost, CatBoost, EBM)
  6. Evaluation: ROC-AUC, PR-AUC, F1, KS, Brier + diagnostic plots; best model selected
  7. Explainability: SHAP on XGBoost (fixed reference model) and on the overall best
     model if different; EBM's native glass-box explanation; SHAP-vs-EBM rank
     agreement (RQ2)
  8. Fairness audit (pre-mitigation) on SEX / AGE_GROUP, for all four tuned models
  9. Bias mitigation via reweighing on SEX + re-train + re-audit, for all four
     model families (RQ3 result = the best model's before/after; RQ4 result =
     whether the same mitigation recipe works comparably across all four)
 10. Persist trained models + a consolidated run_summary.json

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
from dac.data.loader import load_uci_credit
from dac.eda.eda_report import run_eda
from dac.explainability.ebm_explain import compute_shap_ebm_agreement, explain_ebm_model
from dac.explainability.shap_explain import explain_model
from dac.fairness.audit import audit_fairness
from dac.fairness.mitigation import compute_reweighing_weights
from dac.features.engineering import build_preprocessor, engineer_uci_credit_features, split_feature_columns
from dac.models.evaluate import compare_models, compute_metrics, plot_evaluation_suite
from dac.models.train import (
    build_catboost_pipeline,
    build_ebm_pipeline,
    build_logistic_regression_pipeline,
    build_xgboost_pipeline,
    compute_balanced_sample_weight,
    compute_scale_pos_weight,
    fit,
)
from dac.models.tune import tune_catboost, tune_ebm, tune_logistic_regression, tune_xgboost
from dac.utils.logging_utils import get_logger

logger = get_logger("run_pipeline")

MODEL_NAMES = ["logistic_regression", "xgboost", "catboost", "ebm"]


def main(quick: bool = False) -> dict:
    t0 = time.time()
    seed = CONFIG["seed"]
    figures_dir = CONFIG["paths"]["figures_dir"]
    metrics_dir = CONFIG["paths"]["metrics_dir"]
    models_dir = CONFIG["paths"]["models_dir"]

    n_trials = 5 if quick else CONFIG["tuning"]["n_trials"]
    cb_trials = 4 if quick else CONFIG["tuning"]["catboost_n_trials"]
    ebm_trials = 3 if quick else CONFIG["tuning"]["ebm_n_trials"]
    cv_folds = 3 if quick else CONFIG["tuning"]["cv_folds"]
    ebm_cv_folds = 2  # EBM fits are expensive at this row count; 2-fold keeps tuning wall-clock reasonable

    summary: dict = {}

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 1: Load UCI credit-card data\n" + "=" * 70)
    cfg = CONFIG["data"]["uci_credit"]
    df, is_synthetic = load_uci_credit()
    summary["uci_credit_is_synthetic"] = is_synthetic
    summary["uci_credit_shape"] = list(df.shape)

    # ------------------------------------------------------------------
    # 2. Feature engineering (before EDA, so protected attributes derived
    #    here -- AGE_GROUP -- are present for the EDA's protected-attribute
    #    breakdown, not just the original SEX column)
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 2: Feature engineering\n" + "=" * 70)
    df_fe = engineer_uci_credit_features(df)

    # ------------------------------------------------------------------
    # 3. EDA + split
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 3: EDA\n" + "=" * 70)
    eda_stats = run_eda(
        df_fe,
        target_col=cfg["target_col"],
        protected_attributes=cfg["protected_attributes"],
        figures_dir=figures_dir,
        report_path=metrics_dir / "eda_uci_credit.md",
        dataset_name="uci_credit",
    )
    summary["uci_credit_eda"] = eda_stats

    target_col = cfg["target_col"]
    id_col = cfg["id_col"]
    protected_attributes = cfg["protected_attributes"]
    exclude_cols = [id_col, *protected_attributes]
    numeric_cols, categorical_cols = split_feature_columns(df_fe, target_col, exclude_cols)
    logger.info("UCI credit model features: %d numeric, %d categorical", len(numeric_cols), len(categorical_cols))

    X = df_fe[numeric_cols + categorical_cols]
    y = df_fe[target_col]
    sensitive = df_fe[protected_attributes]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=CONFIG["split"]["test_size"], random_state=seed, stratify=y
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    ebm_weights_train = compute_balanced_sample_weight(y_train)

    # ------------------------------------------------------------------
    # 4. Baseline training (all four model families)
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 4: Baseline training\n" + "=" * 70)
    lr_baseline = fit(
        build_logistic_regression_pipeline(preprocessor, max_iter=CONFIG["models"]["logistic_regression"]["max_iter"]),
        X_train, y_train, "logistic_regression_baseline",
    )
    xgb_baseline = fit(
        build_xgboost_pipeline(
            preprocessor, n_estimators=CONFIG["models"]["xgboost"]["n_estimators"],
            scale_pos_weight=compute_scale_pos_weight(y_train),
        ),
        X_train, y_train, "xgboost_baseline",
    )
    cb_baseline = fit(
        build_catboost_pipeline(preprocessor, n_estimators=CONFIG["models"]["catboost"]["n_estimators"]),
        X_train, y_train, "catboost_baseline",
    )
    ebm_baseline = fit(
        build_ebm_pipeline(preprocessor, max_bins=CONFIG["models"]["ebm"]["max_bins"]),
        X_train, y_train, "ebm_baseline", sample_weight=ebm_weights_train,
    )

    baseline_results = {}
    for trained in (lr_baseline, xgb_baseline, cb_baseline, ebm_baseline):
        proba = trained.pipeline.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test.to_numpy(), proba)
        baseline_results[trained.name] = m
        plot_evaluation_suite(y_test.to_numpy(), proba, trained.name, figures_dir / "uci_credit")
    compare_models(baseline_results, metrics_dir / "baseline")
    summary["baseline_metrics"] = baseline_results

    # ------------------------------------------------------------------
    # 5. Hyperparameter tuning
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 5: Hyperparameter tuning\n" + "=" * 70)
    lr_tuned = tune_logistic_regression(preprocessor, X_train, y_train, n_iter=15 if not quick else 4, cv_folds=cv_folds, seed=seed)
    xgb_tuned = tune_xgboost(preprocessor, X_train, y_train, n_trials=n_trials, cv_folds=cv_folds, seed=seed)
    cb_tuned = tune_catboost(preprocessor, X_train, y_train, n_trials=cb_trials, cv_folds=cv_folds, seed=seed)
    ebm_tuned = tune_ebm(preprocessor, X_train, y_train, n_trials=ebm_trials, cv_folds=ebm_cv_folds, seed=seed)

    lr_tuned.pipeline.fit(X_train, y_train)
    xgb_tuned.pipeline.fit(X_train, y_train)
    cb_tuned.pipeline.fit(X_train, y_train)
    ebm_tuned.pipeline.fit(X_train, y_train, clf__sample_weight=ebm_weights_train)

    # ------------------------------------------------------------------
    # 6. Evaluation of tuned models
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 6: Evaluate tuned models\n" + "=" * 70)
    tuned_pipelines = {
        "logistic_regression_tuned": lr_tuned.pipeline,
        "xgboost_tuned": xgb_tuned.pipeline,
        "catboost_tuned": cb_tuned.pipeline,
        "ebm_tuned": ebm_tuned.pipeline,
    }
    tuned_results = {}
    for name, pipeline in tuned_pipelines.items():
        proba = pipeline.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test.to_numpy(), proba)
        tuned_results[name] = m
        plot_evaluation_suite(y_test.to_numpy(), proba, name, figures_dir / "uci_credit")
    all_results = {**baseline_results, **tuned_results}
    compare_models(all_results, metrics_dir)
    summary["tuned_metrics"] = tuned_results
    summary["tuning_best_params"] = {
        "logistic_regression": lr_tuned.best_params,
        "xgboost": xgb_tuned.best_params,
        "catboost": cb_tuned.best_params,
        "ebm": ebm_tuned.best_params,
    }

    best_model_name = max(tuned_results, key=lambda k: tuned_results[k]["roc_auc"])
    best_pipeline = tuned_pipelines[best_model_name]
    logger.info("Best model by ROC-AUC: %s", best_model_name)
    summary["best_model"] = best_model_name

    for name, pipeline in tuned_pipelines.items():
        joblib.dump(pipeline, models_dir / f"{name}.joblib")

    # ------------------------------------------------------------------
    # 7. Explainability: SHAP (XGBoost + best model) + native EBM + agreement
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 7: Explainability (SHAP + EBM)\n" + "=" * 70)
    shap_importance_xgb = explain_model(
        tuned_pipelines["xgboost_tuned"], X_train, X_test, "xgboost_tuned", figures_dir / "uci_credit" / "shap"
    )
    summary["shap_top_features"] = shap_importance_xgb.head(10)["mean_abs_shap"].to_dict()

    if best_model_name != "xgboost_tuned" and best_model_name != "ebm_tuned":
        shap_importance_best = explain_model(
            best_pipeline, X_train, X_test, best_model_name, figures_dir / "uci_credit" / "shap"
        )
        summary["shap_top_features_best_model"] = shap_importance_best.head(10)["mean_abs_shap"].to_dict()

    ebm_importance = explain_ebm_model(tuned_pipelines["ebm_tuned"], "ebm_tuned", figures_dir / "uci_credit" / "ebm")
    summary["ebm_top_features"] = ebm_importance.head(10)["ebm_importance"].to_dict()

    agreement = compute_shap_ebm_agreement(shap_importance_xgb["mean_abs_shap"], ebm_importance["ebm_importance"])
    summary["shap_ebm_agreement"] = agreement
    logger.info(
        "SHAP (XGBoost) vs. EBM native importance rank agreement: Spearman r=%s (n=%s common features)",
        agreement["spearman_r"], agreement["n_common_features"],
    )

    # ------------------------------------------------------------------
    # 8. Fairness audit (pre-mitigation), all four tuned models
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 8: Fairness audit (pre-mitigation)\n" + "=" * 70)
    favorable_label = CONFIG["fairness"]["favorable_label"]
    pre_mitigation_fairness: dict = {}
    for name, pipeline in tuned_pipelines.items():
        y_pred = (pipeline.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        pre_mitigation_fairness[name] = {}
        for attr in protected_attributes:
            pre_mitigation_fairness[name][attr] = audit_fairness(
                y_true=y_test.to_numpy(),
                y_pred=y_pred,
                sensitive_features=sens_test[attr],
                model_name=f"{name}_pre_mitigation",
                attribute_name=attr,
                figures_dir=figures_dir / "uci_credit" / "fairness",
                metrics_dir=metrics_dir / "fairness",
                favorable_label=favorable_label,
            )
    summary["fairness_pre_mitigation"] = pre_mitigation_fairness

    # ------------------------------------------------------------------
    # 9. Bias mitigation (reweighing on SEX) + re-audit, all four models
    #    (RQ3: best model's before/after; RQ4: does the same recipe work
    #    comparably across all four model families?)
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 9: Bias mitigation (reweighing on SEX)\n" + "=" * 70)
    mitigation_attr = CONFIG["fairness"]["mitigation_attribute"]
    reweigh_weights = compute_reweighing_weights(y_train, sens_train[mitigation_attr])

    mitigated_builders = {
        "logistic_regression_tuned": lambda: build_logistic_regression_pipeline(preprocessor),
        "xgboost_tuned": lambda: build_xgboost_pipeline(
            preprocessor, n_estimators=xgb_tuned.best_params.get("n_estimators", 400),
            scale_pos_weight=compute_scale_pos_weight(y_train),
        ),
        "catboost_tuned": lambda: build_catboost_pipeline(
            preprocessor, n_estimators=cb_tuned.best_params.get("n_estimators", 400)
        ),
        "ebm_tuned": lambda: build_ebm_pipeline(preprocessor, max_bins=CONFIG["models"]["ebm"]["max_bins"]),
    }

    mitigated_perf: dict = {}
    post_mitigation_fairness: dict = {}
    for name, builder in mitigated_builders.items():
        mitigated_pipeline = builder()
        mitigated_pipeline.fit(X_train, y_train, clf__sample_weight=reweigh_weights)
        joblib.dump(mitigated_pipeline, models_dir / f"{name}_mitigated.joblib")

        proba_mitigated = mitigated_pipeline.predict_proba(X_test)[:, 1]
        y_pred_mitigated = (proba_mitigated >= 0.5).astype(int)
        mitigated_perf[name] = compute_metrics(y_test.to_numpy(), proba_mitigated)
        plot_evaluation_suite(y_test.to_numpy(), proba_mitigated, f"{name}_mitigated", figures_dir / "uci_credit")

        post_mitigation_fairness[name] = {}
        for attr in protected_attributes:
            post_mitigation_fairness[name][attr] = audit_fairness(
                y_true=y_test.to_numpy(),
                y_pred=y_pred_mitigated,
                sensitive_features=sens_test[attr],
                model_name=f"{name}_post_mitigation",
                attribute_name=attr,
                figures_dir=figures_dir / "uci_credit" / "fairness",
                metrics_dir=metrics_dir / "fairness",
                favorable_label=favorable_label,
            )

        dp_before = pre_mitigation_fairness[name][mitigation_attr]["demographic_parity_ratio"]
        dp_after = post_mitigation_fairness[name][mitigation_attr]["demographic_parity_ratio"]
        logger.info(
            "%s: reweighing effect on %s demographic parity ratio: %.3f -> %.3f; ROC-AUC: %.4f -> %.4f",
            name, mitigation_attr, dp_before, dp_after,
            tuned_results[name]["roc_auc"], mitigated_perf[name]["roc_auc"],
        )

    summary["mitigated_performance"] = mitigated_perf
    summary["fairness_post_mitigation"] = post_mitigation_fairness
    summary["fairness_mitigation_attribute"] = mitigation_attr

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
