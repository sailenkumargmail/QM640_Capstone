"""Baseline model training: Logistic Regression + XGBoost, sharing the same
preprocessing ColumnTransformer via sklearn Pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline  # preprocessor + estimator, fitted


def build_logistic_regression_pipeline(
    preprocessor: ColumnTransformer, max_iter: int = 2000, class_weight: str | dict | None = "balanced"
) -> Pipeline:
    clf = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def build_xgboost_pipeline(
    preprocessor: ColumnTransformer,
    n_estimators: int = 400,
    tree_method: str = "hist",
    scale_pos_weight: float | None = None,
    **kwargs,
) -> Pipeline:
    clf = XGBClassifier(
        n_estimators=n_estimators,
        tree_method=tree_method,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        **kwargs,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    y = np.asarray(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float(n_neg / max(n_pos, 1))


def fit(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, name: str) -> TrainedModel:
    logger.info("Fitting %s on %d rows / %d cols", name, *X_train.shape)
    pipeline.fit(X_train, y_train)
    return TrainedModel(name=name, pipeline=pipeline)
