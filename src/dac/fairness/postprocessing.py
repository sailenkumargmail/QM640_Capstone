"""Bias mitigation via equalized-odds post-processing (Hardt, Price, &
Srebro, 2016), implemented with Fairlearn's ``ThresholdOptimizer``.

This is the study's second bias-mitigation technique, evaluated alongside
Kamiran & Calders (2012) reweighing (``dac.fairness.mitigation``). Unlike
reweighing (a pre-processing technique that reweights training rows so the
protected attribute and label become independent, then retrains the model),
equalized-odds post-processing is a post-processing technique: it leaves an
already-fitted model's scores untouched and instead selects group-specific
decision thresholds that equalize true-positive and false-positive rates
across protected groups. It was chosen as the second technique (over
adversarial debiasing) because it requires no new heavy dependency and no
retraining -- it composes with any already-tuned classifier, keeping the
comparison a fair like-for-like addition to the existing pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


class _Float64ProbaEstimator(BaseEstimator):
    """Wraps a fitted estimator so ``predict_proba`` always returns float64.

    XGBoost/CatBoost return float32 probabilities. Fairlearn's
    ``ThresholdOptimizer``/``InterpolatedThresholder`` builds a pandas Series
    from that array and later assigns float64 interpolated values back into
    it in-place; under pandas >= 2's strict setitem casting this raises
    ``LossySetitemError`` -> ``TypeError`` for a float32 Series. Forcing
    float64 at the source avoids the mismatch without patching Fairlearn or
    downgrading pandas.

    Subclasses ``BaseEstimator`` (with a no-op ``fit`` and a trailing-
    underscore attribute) purely so sklearn's ``check_is_fitted`` -- called
    internally by Fairlearn even in ``prefit=True`` mode -- recognizes this
    wrapper as a fitted estimator without trying to refit the pipeline it wraps.
    """

    def __init__(self, fitted_estimator: BaseEstimator | None = None):
        self.fitted_estimator = fitted_estimator
        # Already fitted (it wraps an already-fitted pipeline) -- set the
        # trailing-underscore attribute check_is_fitted looks for up front,
        # since Fairlearn's prefit=True path never calls .fit() on us.
        self.is_fitted_ = True

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        return np.asarray(self.fitted_estimator.predict_proba(X), dtype=np.float64)

    def predict(self, X):
        return np.asarray(self.fitted_estimator.predict(X))


def fit_equalized_odds_postprocessor(
    fitted_pipeline: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_train: pd.Series,
) -> ThresholdOptimizer:
    """Fit a ThresholdOptimizer on top of an already-fitted pipeline.

    ``prefit=True`` tells Fairlearn not to refit the wrapped estimator -- it
    only learns group-specific thresholds from the pipeline's existing
    predictions on the training set.
    """
    postprocessor = ThresholdOptimizer(
        estimator=_Float64ProbaEstimator(fitted_pipeline),
        constraints="equalized_odds",
        objective="accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    postprocessor.fit(X_train, y_train, sensitive_features=sensitive_train)
    logger.info("Fitted equalized-odds ThresholdOptimizer on top of %r", type(fitted_pipeline).__name__)
    return postprocessor


def predict_equalized_odds(
    postprocessor: ThresholdOptimizer,
    X: pd.DataFrame,
    sensitive_features: pd.Series,
    seed: int = 42,
) -> np.ndarray:
    """Group-specific-threshold predictions (0/1). ThresholdOptimizer
    randomizes between two thresholds per group to satisfy the equalized-odds
    constraint exactly, so a fixed random_state is required for reproducibility.
    """
    return np.asarray(
        postprocessor.predict(X, sensitive_features=sensitive_features, random_state=seed)
    )
