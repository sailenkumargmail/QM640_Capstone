import numpy as np
from sklearn.model_selection import train_test_split

from dac.data.synthetic import generate_uci_credit_synthetic
from dac.features.engineering import build_preprocessor, engineer_uci_credit_features, split_feature_columns
from dac.models.evaluate import compute_metrics, ks_statistic
from dac.models.train import (
    build_catboost_pipeline,
    build_ebm_pipeline,
    build_logistic_regression_pipeline,
    build_xgboost_pipeline,
    compute_balanced_sample_weight,
    compute_scale_pos_weight,
    fit,
)


def _prepare_split(n_rows=1500, seed=1):
    df = generate_uci_credit_synthetic(n_rows=n_rows, seed=seed)
    df = engineer_uci_credit_features(df)
    numeric_cols, categorical_cols = split_feature_columns(
        df, target_col="DEFAULT_PAYMENT_NEXT_MONTH", exclude_cols=["ID", "SEX", "AGE_GROUP"]
    )
    X = df[numeric_cols + categorical_cols]
    y = df["DEFAULT_PAYMENT_NEXT_MONTH"]
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y), numeric_cols, categorical_cols


def test_logistic_regression_pipeline_trains_and_predicts():
    (X_train, X_test, y_train, y_test), numeric_cols, categorical_cols = _prepare_split()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = build_logistic_regression_pipeline(preprocessor)
    trained = fit(pipeline, X_train, y_train, "lr_test")
    proba = trained.pipeline.predict_proba(X_test)[:, 1]
    assert proba.shape[0] == len(X_test)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_xgboost_pipeline_trains_and_predicts():
    (X_train, X_test, y_train, y_test), numeric_cols, categorical_cols = _prepare_split()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = build_xgboost_pipeline(preprocessor, n_estimators=50, scale_pos_weight=compute_scale_pos_weight(y_train))
    trained = fit(pipeline, X_train, y_train, "xgb_test")
    proba = trained.pipeline.predict_proba(X_test)[:, 1]
    assert proba.shape[0] == len(X_test)


def test_catboost_pipeline_trains_and_predicts():
    (X_train, X_test, y_train, y_test), numeric_cols, categorical_cols = _prepare_split()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = build_catboost_pipeline(preprocessor, n_estimators=50)
    trained = fit(pipeline, X_train, y_train, "catboost_test")
    proba = trained.pipeline.predict_proba(X_test)[:, 1]
    assert proba.shape[0] == len(X_test)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_ebm_pipeline_trains_and_predicts_with_sample_weight():
    (X_train, X_test, y_train, y_test), numeric_cols, categorical_cols = _prepare_split()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = build_ebm_pipeline(preprocessor, max_bins=64, outer_bags=4)
    weights = compute_balanced_sample_weight(y_train)
    trained = fit(pipeline, X_train, y_train, "ebm_test", sample_weight=weights)
    proba = trained.pipeline.predict_proba(X_test)[:, 1]
    assert proba.shape[0] == len(X_test)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_compute_metrics_keys_and_ranges():
    (X_train, X_test, y_train, y_test), numeric_cols, categorical_cols = _prepare_split()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = build_logistic_regression_pipeline(preprocessor)
    trained = fit(pipeline, X_train, y_train, "lr_test")
    proba = trained.pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), proba)
    for key in ["roc_auc", "pr_auc", "f1", "ks_statistic", "brier_score"]:
        assert key in metrics
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["brier_score"] <= 1


def test_ks_statistic_perfect_separation_is_one():
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.array([0.0] * 50 + [1.0] * 50)
    assert ks_statistic(y_true, y_proba) == 1.0
