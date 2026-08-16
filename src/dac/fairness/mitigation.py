"""Bias mitigation via reweighing (Kamiran & Calders, 2012).

Pre-processing technique: compute a per-row sample weight so that, in the
weighted training set, the protected attribute and the label are
statistically independent. Weight for a row with group g and label l:

    w(g, l) = ( P(group=g) * P(label=l) ) / P(group=g, label=l)

This is the standard Kamiran-Calders reweighing formula, chosen because it
is model-agnostic (works identically for Logistic Regression and XGBoost)
and requires no change to the model architecture -- a pragmatic MVP choice
per the reviewer critique's recommendation to scope down from the original
three-mitigation-technique matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_reweighing_weights(y: pd.Series, sensitive_feature: pd.Series) -> np.ndarray:
    df = pd.DataFrame({"y": np.asarray(y), "g": np.asarray(sensitive_feature)})
    n = len(df)

    p_group = df["g"].value_counts(normalize=True)
    p_label = df["y"].value_counts(normalize=True)
    p_joint = df.groupby(["g", "y"]).size() / n

    weights = np.empty(n)
    for i, (g, y_val) in enumerate(zip(df["g"], df["y"])):
        joint = p_joint.get((g, y_val), np.nan)
        weights[i] = (p_group[g] * p_label[y_val]) / joint if joint and joint > 0 else 1.0

    weights = weights / weights.mean()  # normalize so mean weight == 1
    logger.info(
        "Reweighing: weight range [%.3f, %.3f], mean=%.3f",
        weights.min(), weights.max(), weights.mean(),
    )
    return weights


def build_joint_group(df: pd.DataFrame, attrs: list[str]) -> pd.Series:
    """Concatenate two or more protected-attribute columns into a single
    joint-group label (e.g. SEX x AGE_GROUP -> "1_25-34"), so the existing
    single-attribute ``compute_reweighing_weights`` can be reused unchanged
    to reweigh on the intersection of several attributes at once (a joint
    sensitivity analysis, vs. mitigating on one attribute at a time).
    """
    if len(attrs) < 2:
        raise ValueError("build_joint_group needs at least 2 attributes")
    joint = df[attrs[0]].astype(str)
    for attr in attrs[1:]:
        joint = joint + "_" + df[attr].astype(str)
    joint.name = "_".join(attrs)
    return joint
