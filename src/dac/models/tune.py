"""Hyperparameter tuning.

Logistic Regression: RandomizedSearchCV over C / penalty / solver.
XGBoost: Optuna Bayesian search (TPE) with stratified CV, optimizing ROC-AUC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from dac.models.train import compute_balanced_sample_weight, compute_scale_pos_weight
from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    name: str
    best_params: dict
    best_cv_score: float
    pipeline: Pipeline  # best pipeline, NOT yet refit on full train set


def tune_logistic_regression(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    seed: int = 42,
) -> TuningResult:
    logger.info("Tuning Logistic Regression (RandomizedSearchCV, n_iter=%d)", n_iter)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    param_distributions = {
        "clf__C": np.logspace(-3, 2, 50),
        "clf__solver": ["lbfgs", "liblinear"],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info("Best LR params: %s | CV %s=%.4f", search.best_params_, scoring, search.best_score_)
    return TuningResult(
        name="logistic_regression",
        best_params=search.best_params_,
        best_cv_score=float(search.best_score_),
        pipeline=search.best_estimator_,
    )


def tune_xgboost(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 25,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    seed: int = 42,
) -> TuningResult:
    logger.info("Tuning XGBoost (Optuna, n_trials=%d)", n_trials)
    scale_pos_weight = compute_scale_pos_weight(y_train)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        clf = XGBClassifier(
            **params,
            tree_method="hist",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=-1,
        )
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
        scores = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Best XGB params: %s | CV %s=%.4f", study.best_params, scoring, study.best_value)

    best_clf = XGBClassifier(
        **study.best_params,
        tree_method="hist",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
    )
    best_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", best_clf)])

    return TuningResult(
        name="xgboost",
        best_params=study.best_params,
        best_cv_score=float(study.best_value),
        pipeline=best_pipeline,
    )


def tune_catboost(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 20,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    seed: int = 42,
) -> TuningResult:
    logger.info("Tuning CatBoost (Optuna, n_trials=%d)", n_trials)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "depth": trial.suggest_int("depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        clf = CatBoostClassifier(
            **params,
            auto_class_weights="Balanced",
            bootstrap_type="Bernoulli",
            random_state=seed,
            verbose=False,
        )
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
        scores = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Best CatBoost params: %s | CV %s=%.4f", study.best_params, scoring, study.best_value)

    best_clf = CatBoostClassifier(
        **study.best_params,
        auto_class_weights="Balanced",
        bootstrap_type="Bernoulli",
        random_state=seed,
        verbose=False,
    )
    best_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", best_clf)])

    return TuningResult(
        name="catboost",
        best_params=study.best_params,
        best_cv_score=float(study.best_value),
        pipeline=best_pipeline,
    )


def tune_ebm(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 12,
    cv_folds: int = 3,
    scoring: str = "roc_auc",
    seed: int = 42,
) -> TuningResult:
    """EBM fits are considerably slower per-trial than XGBoost/CatBoost at
    this row count, hence the smaller default trial count and CV-fold count.
    """
    logger.info("Tuning EBM (Optuna, n_trials=%d)", n_trials)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_bins": trial.suggest_categorical("max_bins", [128, 256, 512]),
            "max_leaves": trial.suggest_int("max_leaves", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "outer_bags": trial.suggest_int("outer_bags", 2, 4),
        }
        # n_jobs=1: repeated multiprocessing-pool spin-up per fit dominates
        # wall-clock at this row count far more than sequential outer bags
        # would (see dac.models.train.build_ebm_pipeline).
        clf = ExplainableBoostingClassifier(**params, random_state=seed, n_jobs=1)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
        scores = cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Best EBM params: %s | CV %s=%.4f", study.best_params, scoring, study.best_value)

    best_clf = ExplainableBoostingClassifier(**study.best_params, random_state=seed, n_jobs=1)
    best_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", best_clf)])

    return TuningResult(
        name="ebm",
        best_params=study.best_params,
        best_cv_score=float(study.best_value),
        pipeline=best_pipeline,
    )
