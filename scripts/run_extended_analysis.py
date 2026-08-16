"""Extended analysis closing the four gaps flagged as "Pending" in the
Interim Report, for the Final Report:

  1. Exact hypothesis-test p-values for RQ1 (bootstrap AUC test, best tuned
     model vs. tuned Logistic Regression) and RQ3/RQ4 (two-proportion
     z-tests on the favorable-outcome selection-rate gap, pre- and
     post-mitigation, per tuned model, per protected attribute).
  2. A joint SEX x AGE_GROUP reweighing sensitivity run for the best model,
     compared against the existing SEX-only mitigation.
  3. A second bias-mitigation technique -- equalized-odds post-processing
     (Fairlearn ThresholdOptimizer) -- for the best model on SEX, compared
     against reweighing.
  4. An accuracy-fairness decision framework comparing all of the above.

This script deliberately does NOT re-run hyperparameter tuning (the full
scripts/run_pipeline.py takes ~39 minutes, dominated by Optuna search). It
reloads the already-tuned model artifacts in models/*.joblib and
reconstructs the identical (seeded, stratified) train/test split, so it
runs in well under a minute. A sanity check asserts the reloaded split
reproduces run_pipeline.py's own recorded ROC-AUC before anything else runs.

Usage:
    python scripts/run_extended_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dac.config import CONFIG
from dac.data.loader import load_uci_credit
from dac.fairness.audit import audit_fairness
from dac.fairness.mitigation import build_joint_group, compute_reweighing_weights
from dac.fairness.postprocessing import fit_equalized_odds_postprocessor, predict_equalized_odds
from dac.fairness.significance import bootstrap_auc_test, two_proportion_ztest
from dac.features.engineering import build_preprocessor, engineer_uci_credit_features, split_feature_columns
from dac.models.evaluate import compute_metrics
from dac.models.train import (
    build_catboost_pipeline,
    build_ebm_pipeline,
    build_logistic_regression_pipeline,
    build_xgboost_pipeline,
    compute_scale_pos_weight,
    fit,
)
from dac.utils.logging_utils import get_logger

logger = get_logger("run_extended_analysis")

MODEL_NAMES = ["logistic_regression_tuned", "xgboost_tuned", "catboost_tuned", "ebm_tuned"]
FAMILY_OF = {
    "logistic_regression_tuned": "logistic_regression",
    "xgboost_tuned": "xgboost",
    "catboost_tuned": "catboost",
    "ebm_tuned": "ebm",
}


def build_candidate_pipeline(model_name: str, preprocessor, y_train, tuning_best_params: dict):
    """Mirrors scripts/run_pipeline.py's `mitigated_builders` dict exactly,
    so a mitigated retrain (joint-attribute reweighing here; SEX-only
    reweighing there) is an apples-to-apples comparison -- same
    hyperparameters, different sample weights.
    """
    if model_name == "logistic_regression_tuned":
        return build_logistic_regression_pipeline(preprocessor)
    if model_name == "xgboost_tuned":
        return build_xgboost_pipeline(
            preprocessor,
            n_estimators=tuning_best_params["xgboost"].get("n_estimators", 400),
            scale_pos_weight=compute_scale_pos_weight(y_train),
        )
    if model_name == "catboost_tuned":
        return build_catboost_pipeline(
            preprocessor, n_estimators=tuning_best_params["catboost"].get("n_estimators", 400)
        )
    if model_name == "ebm_tuned":
        return build_ebm_pipeline(preprocessor, max_bins=CONFIG["models"]["ebm"]["max_bins"])
    raise ValueError(f"unknown model_name: {model_name}")


def selection_counts_by_group(y_pred: np.ndarray, groups: pd.Series, favorable_label: int) -> dict:
    favorable = (y_pred == favorable_label).astype(int)
    df = pd.DataFrame({"g": np.asarray(groups), "fav": favorable})
    counts = df.groupby("g")["fav"].agg(["sum", "count"])
    return {g: {"count": int(row["sum"]), "n": int(row["count"])} for g, row in counts.iterrows()}


def extreme_group_pair(counts: dict) -> tuple[str, str]:
    """(max-rate group, min-rate group) -- the pair driving the demographic
    parity ratio (which is itself min-rate / max-rate across all groups)."""
    rates = {g: v["count"] / v["n"] for g, v in counts.items()}
    g_max = max(rates, key=rates.get)
    g_min = min(rates, key=rates.get)
    return g_max, g_min


def main() -> dict:
    metrics_dir = CONFIG["paths"]["metrics_dir"]
    figures_dir = CONFIG["paths"]["figures_dir"]
    models_dir = CONFIG["paths"]["models_dir"]
    seed = CONFIG["seed"]

    sig_dir = metrics_dir / "significance"
    sens_dir = metrics_dir / "sensitivity"
    mit2_dir = metrics_dir / "mitigation2_equalized_odds"
    decision_fig_dir = figures_dir / "uci_credit" / "decision_framework"
    for d in (sig_dir, sens_dir, mit2_dir, decision_fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_summary = json.loads((metrics_dir / "run_summary.json").read_text(encoding="utf-8"))
    tuned_metrics = run_summary["tuned_metrics"]
    tuning_best_params = run_summary["tuning_best_params"]
    best_model_name = run_summary["best_model"]
    favorable_label = CONFIG["fairness"]["favorable_label"]
    mitigation_attr = run_summary["fairness_mitigation_attribute"]

    logger.info("=" * 70 + "\nSTEP 0: Reconstruct data split (must match run_pipeline.py exactly)\n" + "=" * 70)
    cfg = CONFIG["data"]["uci_credit"]
    df, _ = load_uci_credit()
    df_fe = engineer_uci_credit_features(df)

    target_col = cfg["target_col"]
    id_col = cfg["id_col"]
    protected_attributes = cfg["protected_attributes"]
    exclude_cols = [id_col, *protected_attributes]
    numeric_cols, categorical_cols = split_feature_columns(df_fe, target_col, exclude_cols)

    X = df_fe[numeric_cols + categorical_cols]
    y = df_fe[target_col]
    sensitive = df_fe[protected_attributes]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=CONFIG["split"]["test_size"], random_state=seed, stratify=y
    )

    tuned_pipelines = {name: joblib.load(models_dir / f"{name}.joblib") for name in MODEL_NAMES}
    mitigated_pipelines = {name: joblib.load(models_dir / f"{name}_mitigated.joblib") for name in MODEL_NAMES}

    # Sanity check: split reconstruction must reproduce the pipeline's own recorded metrics.
    sanity_proba = tuned_pipelines[best_model_name].predict_proba(X_test)[:, 1]
    sanity_auc = roc_auc_score(y_test, sanity_proba)
    expected_auc = tuned_metrics[best_model_name]["roc_auc"]
    if abs(sanity_auc - expected_auc) > 1e-6:
        raise RuntimeError(
            f"Split reconstruction mismatch: recomputed ROC-AUC={sanity_auc:.6f} vs "
            f"run_summary.json={expected_auc:.6f}. The train/test split is not reproducing "
            "the original run -- do not trust downstream results until this is fixed."
        )
    logger.info("Sanity check passed: recomputed %s ROC-AUC = %.6f matches run_summary.json", best_model_name, sanity_auc)

    results: dict = {"best_model": best_model_name, "mitigation_attribute": mitigation_attr}

    # ------------------------------------------------------------------
    # RQ1: bootstrap AUC significance test, best model vs tuned LR
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 1: RQ1 -- bootstrap AUC test (best model vs tuned LR)\n" + "=" * 70)
    proba_best = tuned_pipelines[best_model_name].predict_proba(X_test)[:, 1]
    proba_lr = tuned_pipelines["logistic_regression_tuned"].predict_proba(X_test)[:, 1]
    rq1 = bootstrap_auc_test(y_test.to_numpy(), proba_best, proba_lr, n_boot=10000, seed=seed, alternative="larger")
    rq1["model_a"] = best_model_name
    rq1["model_b"] = "logistic_regression_tuned"
    (sig_dir / "rq1_auc_bootstrap.json").write_text(json.dumps(rq1, indent=2), encoding="utf-8")
    results["rq1_significance"] = rq1

    # ------------------------------------------------------------------
    # RQ3/RQ4: two-proportion z-tests, pre- and post-mitigation, per model per attribute
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 2: RQ3/RQ4 -- two-proportion z-tests (pre/post mitigation)\n" + "=" * 70)
    rq3_rq4: dict = {}
    for name in MODEL_NAMES:
        rq3_rq4[name] = {}
        y_pred_pre = (tuned_pipelines[name].predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        y_pred_post = (mitigated_pipelines[name].predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        for attr in protected_attributes:
            counts_pre = selection_counts_by_group(y_pred_pre, sens_test[attr], favorable_label)
            counts_post = selection_counts_by_group(y_pred_post, sens_test[attr], favorable_label)

            g_max_pre, g_min_pre = extreme_group_pair(counts_pre)
            z_pre = two_proportion_ztest(
                counts_pre[g_max_pre]["count"], counts_pre[g_max_pre]["n"],
                counts_pre[g_min_pre]["count"], counts_pre[g_min_pre]["n"],
                alternative="two-sided",
            )
            z_pre["group_max"], z_pre["group_min"] = str(g_max_pre), str(g_min_pre)

            g_max_post, g_min_post = extreme_group_pair(counts_post)
            z_post = two_proportion_ztest(
                counts_post[g_max_post]["count"], counts_post[g_max_post]["n"],
                counts_post[g_min_post]["count"], counts_post[g_min_post]["n"],
                alternative="two-sided",
            )
            z_post["group_max"], z_post["group_min"] = str(g_max_post), str(g_min_post)

            rq3_rq4[name][attr] = {"pre_mitigation": z_pre, "post_mitigation": z_post}
            logger.info(
                "%s / %s: pre-mitigation gap p=%.4g (%.3f vs %.3f) -> post-mitigation gap p=%.4g (%.3f vs %.3f)",
                name, attr, z_pre["p_value"], z_pre["p1"], z_pre["p2"],
                z_post["p_value"], z_post["p1"], z_post["p2"],
            )
    (sig_dir / "rq3_rq4_ztests.json").write_text(json.dumps(rq3_rq4, indent=2), encoding="utf-8")
    results["rq3_rq4_significance"] = rq3_rq4

    # ------------------------------------------------------------------
    # Joint SEX x AGE_GROUP mitigation sensitivity analysis (best model)
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 3: Joint SEX x AGE_GROUP mitigation sensitivity (%s)\n" + "=" * 70, best_model_name)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    joint_train = build_joint_group(sens_train, ["SEX", "AGE_GROUP"])
    joint_weights = compute_reweighing_weights(y_train, joint_train)

    joint_pipeline = build_candidate_pipeline(best_model_name, preprocessor, y_train, tuning_best_params)
    joint_trained = fit(joint_pipeline, X_train, y_train, f"{best_model_name}_joint_mitigated", sample_weight=joint_weights)
    joblib.dump(joint_trained.pipeline, models_dir / f"{best_model_name}_joint_mitigated.joblib")

    proba_joint = joint_trained.pipeline.predict_proba(X_test)[:, 1]
    y_pred_joint = (proba_joint >= 0.5).astype(int)
    joint_perf = compute_metrics(y_test.to_numpy(), proba_joint)

    joint_fairness: dict = {}
    for attr in protected_attributes:
        joint_fairness[attr] = audit_fairness(
            y_true=y_test.to_numpy(), y_pred=y_pred_joint, sensitive_features=sens_test[attr],
            model_name=f"{best_model_name}_joint_mitigation", attribute_name=attr,
            figures_dir=figures_dir / "uci_credit" / "fairness", metrics_dir=metrics_dir / "fairness",
            favorable_label=favorable_label,
        )

    single_attr_fairness = run_summary["fairness_post_mitigation"][best_model_name]
    joint_comparison = {
        "performance_tuned": tuned_metrics[best_model_name],
        "performance_single_attr_mitigation": run_summary["mitigated_performance"][best_model_name],
        "performance_joint_mitigation": joint_perf,
        "fairness_single_attr_mitigation": {a: single_attr_fairness[a] for a in protected_attributes},
        "fairness_joint_mitigation": joint_fairness,
    }
    (sens_dir / f"{best_model_name}_joint_SEX_AGE_GROUP.json").write_text(
        json.dumps(joint_comparison, indent=2), encoding="utf-8"
    )
    results["joint_mitigation_sensitivity"] = joint_comparison
    logger.info(
        "Joint mitigation: ROC-AUC %.4f -> %.4f; DP ratio SEX %.3f -> %.3f; DP ratio AGE_GROUP %.3f -> %.3f",
        tuned_metrics[best_model_name]["roc_auc"], joint_perf["roc_auc"],
        single_attr_fairness["SEX"]["demographic_parity_ratio"], joint_fairness["SEX"]["demographic_parity_ratio"],
        single_attr_fairness["AGE_GROUP"]["demographic_parity_ratio"], joint_fairness["AGE_GROUP"]["demographic_parity_ratio"],
    )

    # ------------------------------------------------------------------
    # Second mitigation technique: equalized-odds post-processing (best model, SEX)
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 4: Equalized-odds post-processing (%s, SEX)\n" + "=" * 70, best_model_name)
    eo_postprocessor = fit_equalized_odds_postprocessor(
        tuned_pipelines[best_model_name], X_train, y_train, sens_train["SEX"]
    )
    y_pred_eo = predict_equalized_odds(eo_postprocessor, X_test, sens_test["SEX"], seed=seed)

    eo_perf = {
        "accuracy": float(accuracy_score(y_test, y_pred_eo)),
        "f1": float(f1_score(y_test, y_pred_eo)),
        "roc_auc_underlying_model": tuned_metrics[best_model_name]["roc_auc"],
        "note": "ThresholdOptimizer outputs hard group-thresholded labels, not scores, "
                "so ROC-AUC/PR-AUC/Brier are not defined for its output; the underlying "
                "model's (unchanged) ranking ROC-AUC is reported for context only.",
    }
    eo_fairness: dict = {}
    for attr in protected_attributes:
        eo_fairness[attr] = audit_fairness(
            y_true=y_test.to_numpy(), y_pred=y_pred_eo, sensitive_features=sens_test[attr],
            model_name=f"{best_model_name}_eo_postprocessed", attribute_name=attr,
            figures_dir=figures_dir / "uci_credit" / "fairness", metrics_dir=metrics_dir / "fairness",
            favorable_label=favorable_label,
        )
    eo_comparison = {
        "performance": eo_perf,
        "fairness": eo_fairness,
        "performance_reweighing_comparison": run_summary["mitigated_performance"][best_model_name],
        "fairness_reweighing_comparison": {a: single_attr_fairness[a] for a in protected_attributes},
    }
    (mit2_dir / f"{best_model_name}_SEX_equalized_odds.json").write_text(
        json.dumps(eo_comparison, indent=2), encoding="utf-8"
    )
    results["equalized_odds_postprocessing"] = eo_comparison
    logger.info(
        "Equalized-odds postprocessing: F1=%.4f; DP ratio SEX %.3f (reweighing: %.3f)",
        eo_perf["f1"], eo_fairness["SEX"]["demographic_parity_ratio"],
        single_attr_fairness["SEX"]["demographic_parity_ratio"],
    )

    # ------------------------------------------------------------------
    # Decision framework: accuracy vs. fairness across all techniques
    # ------------------------------------------------------------------
    logger.info("=" * 70 + "\nSTEP 5: Accuracy-fairness decision framework\n" + "=" * 70)
    dp_threshold = CONFIG["fairness"]["disparate_impact_threshold"]
    rows = {
        "tuned (no mitigation)": {
            "roc_auc": tuned_metrics[best_model_name]["roc_auc"],
            "f1": tuned_metrics[best_model_name]["f1"],
            "dp_ratio_SEX": run_summary["fairness_pre_mitigation"][best_model_name]["SEX"]["demographic_parity_ratio"],
            "dp_ratio_AGE_GROUP": run_summary["fairness_pre_mitigation"][best_model_name]["AGE_GROUP"]["demographic_parity_ratio"],
            "eo_diff_SEX": run_summary["fairness_pre_mitigation"][best_model_name]["SEX"]["equalized_odds_difference"],
            "eo_diff_AGE_GROUP": run_summary["fairness_pre_mitigation"][best_model_name]["AGE_GROUP"]["equalized_odds_difference"],
        },
        "SEX-only reweighing": {
            "roc_auc": run_summary["mitigated_performance"][best_model_name]["roc_auc"],
            "f1": run_summary["mitigated_performance"][best_model_name]["f1"],
            "dp_ratio_SEX": single_attr_fairness["SEX"]["demographic_parity_ratio"],
            "dp_ratio_AGE_GROUP": single_attr_fairness["AGE_GROUP"]["demographic_parity_ratio"],
            "eo_diff_SEX": single_attr_fairness["SEX"]["equalized_odds_difference"],
            "eo_diff_AGE_GROUP": single_attr_fairness["AGE_GROUP"]["equalized_odds_difference"],
        },
        "Joint SEX x AGE_GROUP reweighing": {
            "roc_auc": joint_perf["roc_auc"],
            "f1": joint_perf["f1"],
            "dp_ratio_SEX": joint_fairness["SEX"]["demographic_parity_ratio"],
            "dp_ratio_AGE_GROUP": joint_fairness["AGE_GROUP"]["demographic_parity_ratio"],
            "eo_diff_SEX": joint_fairness["SEX"]["equalized_odds_difference"],
            "eo_diff_AGE_GROUP": joint_fairness["AGE_GROUP"]["equalized_odds_difference"],
        },
        "SEX equalized-odds postprocessing": {
            "roc_auc": np.nan,  # not defined for hard-thresholded output; see eo_perf note
            "f1": eo_perf["f1"],
            "dp_ratio_SEX": eo_fairness["SEX"]["demographic_parity_ratio"],
            "dp_ratio_AGE_GROUP": eo_fairness["AGE_GROUP"]["demographic_parity_ratio"],
            "eo_diff_SEX": eo_fairness["SEX"]["equalized_odds_difference"],
            "eo_diff_AGE_GROUP": eo_fairness["AGE_GROUP"]["equalized_odds_difference"],
        },
    }
    decision_df = pd.DataFrame(rows).T
    decision_df.to_csv(metrics_dir / "decision_framework.csv")
    decision_df.to_json(metrics_dir / "decision_framework.json", orient="index", indent=2)

    # Decision rule computed from the numbers: among techniques that meet the
    # four-fifths threshold on BOTH protected attributes, prefer the one with
    # the highest ROC-AUC (ties broken by lower max equalized-odds difference).
    meets_threshold = decision_df[
        (decision_df["dp_ratio_SEX"] >= dp_threshold) & (decision_df["dp_ratio_AGE_GROUP"] >= dp_threshold)
    ]
    if len(meets_threshold) > 0 and meets_threshold["roc_auc"].notna().any():
        recommended = meets_threshold["roc_auc"].idxmax()
        narrative = (
            f"Among the techniques evaluated, {len(meets_threshold)} of {len(decision_df)} "
            f"meet the four-fifths rule (demographic parity ratio >= {dp_threshold:.2f}) on both "
            f"SEX and AGE_GROUP: {', '.join(meets_threshold.index)}. Of those, "
            f"'{recommended}' retains the highest ROC-AUC "
            f"({decision_df.loc[recommended, 'roc_auc']:.4f}), making it the recommended "
            f"operating point when both a fairness floor and predictive performance matter."
        )
    else:
        narrative = (
            "No technique with a defined ROC-AUC meets the four-fifths rule on both protected "
            "attributes simultaneously; the fairness/accuracy trade-off must be decided "
            "case-by-case based on which attribute's disparity is of greater regulatory concern."
        )
    (metrics_dir / "decision_framework_narrative.md").write_text(narrative, encoding="utf-8")
    logger.info("Decision framework narrative: %s", narrative)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, (label, row) in enumerate(rows.items()):
        if np.isnan(row["roc_auc"]):
            continue
        ax.scatter(row["roc_auc"], min(row["dp_ratio_SEX"], row["dp_ratio_AGE_GROUP"]), s=90, color=palette[i % len(palette)], label=label)
    ax.axhline(dp_threshold, color="red", linestyle="--", label=f"four-fifths threshold ({dp_threshold:.2f})")
    ax.set_xlabel("ROC-AUC (predictive performance)")
    ax.set_ylabel("min(DP ratio SEX, DP ratio AGE_GROUP)")
    ax.set_title(f"{best_model_name}: accuracy-fairness trade-off across mitigation techniques")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(decision_fig_dir / "accuracy_fairness_tradeoff.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    results["decision_framework"] = {
        "table": json.loads(decision_df.to_json(orient="index")),
        "narrative": narrative,
        "four_fifths_threshold": dp_threshold,
    }

    # ------------------------------------------------------------------
    # Wrap up
    # ------------------------------------------------------------------
    (metrics_dir / "extended_summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Extended analysis complete. Summary: %s", metrics_dir / "extended_summary.json")
    return results


if __name__ == "__main__":
    main()
