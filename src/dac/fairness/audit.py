"""Fairness auditing with Fairlearn: per-group selection rates, demographic
parity, equalized odds, and the four-fifths (disparate impact) rule.

Note on the four-fifths rule: this 0.80 disparate-impact-ratio threshold is
an EEOC employment-discrimination heuristic (US), not a codified lending
standard under ECOA/Reg B, and has no direct legal standing outside the US.
It is used here only as a common, borrowed convention from the fairness-ML
literature for flagging practically significant group differences -- not
implied to be a regulatory threshold. (See the mentor/reviewer critique's
"Evaluation Metrics" section.)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)

DISPARATE_IMPACT_THRESHOLD = 0.80


def audit_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    model_name: str,
    attribute_name: str,
    figures_dir: Path,
    metrics_dir: Path,
    favorable_label: int = 0,
) -> dict:
    """favorable_label: which class of y_pred counts as the "favorable"
    outcome for selection-rate purposes (0 = repaid / not-default for the
    UCI DEFAULT_PAYMENT_NEXT_MONTH target).
    """
    figures_dir = Path(figures_dir)
    metrics_dir = Path(metrics_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Fairlearn's selection_rate treats y_pred==1 as "selected"; flip if the
    # favorable outcome is encoded as 0.
    y_pred_favorable = (y_pred == favorable_label).astype(int)
    y_true_favorable = (y_true == favorable_label).astype(int)

    metric_frame = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "accuracy": accuracy_score,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },
        y_true=y_true_favorable,
        y_pred=y_pred_favorable,
        sensitive_features=sensitive_features,
    )

    by_group = metric_frame.by_group
    logger.info("%s / %s fairness by group:\n%s", model_name, attribute_name, by_group.round(4).to_string())

    dp_diff = demographic_parity_difference(y_true_favorable, y_pred_favorable, sensitive_features=sensitive_features)
    dp_ratio = demographic_parity_ratio(y_true_favorable, y_pred_favorable, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true_favorable, y_pred_favorable, sensitive_features=sensitive_features)

    disparate_impact_flag = bool(dp_ratio < DISPARATE_IMPACT_THRESHOLD)

    summary = {
        "model": model_name,
        "attribute": attribute_name,
        "demographic_parity_difference": float(dp_diff),
        "demographic_parity_ratio": float(dp_ratio),
        "equalized_odds_difference": float(eo_diff),
        "four_fifths_rule_threshold": DISPARATE_IMPACT_THRESHOLD,
        "disparate_impact_flag": disparate_impact_flag,
        "by_group": by_group.round(6).to_dict(orient="index"),
    }

    out_stub = metrics_dir / f"fairness_{model_name}_{attribute_name}"
    (out_stub.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    by_group.to_csv(out_stub.with_suffix(".csv"))

    fig, ax = plt.subplots(figsize=(7, 4))
    by_group["selection_rate"].plot(kind="bar", ax=ax, color="#4C72B0")
    ax.axhline(
        by_group["selection_rate"].max() * DISPARATE_IMPACT_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"{DISPARATE_IMPACT_THRESHOLD:.0%} of max group (four-fifths convention)",
    )
    ax.set_title(f"{model_name}: favorable-outcome selection rate by {attribute_name}")
    ax.set_ylabel("selection rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"{model_name}_{attribute_name}_selection_rate.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    if disparate_impact_flag:
        logger.warning(
            "%s / %s: demographic parity ratio %.3f < %.2f -- four-fifths convention flags this as a "
            "practically significant gap.",
            model_name, attribute_name, dp_ratio, DISPARATE_IMPACT_THRESHOLD,
        )

    return summary
