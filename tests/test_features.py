import numpy as np

from dac.data.synthetic import generate_home_credit_synthetic
from dac.features.engineering import (
    build_preprocessor,
    engineer_home_credit_features,
    split_feature_columns,
)


def test_engineer_home_credit_features_adds_columns():
    df = generate_home_credit_synthetic(n_rows=300, seed=1)
    out = engineer_home_credit_features(df)
    for col in ["AGE_YEARS", "YEARS_EMPLOYED", "CREDIT_INCOME_RATIO", "DAYS_EMPLOYED_ANOM", "EXT_SOURCE_MEAN"]:
        assert col in out.columns
    assert (out["AGE_YEARS"] > 0).all()


def test_days_employed_anomaly_handling():
    df = generate_home_credit_synthetic(n_rows=2000, seed=1)
    out = engineer_home_credit_features(df)
    anom_rows = out[out["DAYS_EMPLOYED_ANOM"] == 1]
    if len(anom_rows):
        assert anom_rows["YEARS_EMPLOYED"].isna().all()


def test_split_feature_columns_excludes_target_and_ids():
    df = generate_home_credit_synthetic(n_rows=200, seed=1)
    df = engineer_home_credit_features(df)
    numeric_cols, categorical_cols = split_feature_columns(
        df, target_col="TARGET", exclude_cols=["SK_ID_CURR", "CODE_GENDER", "AGE_GROUP"]
    )
    all_cols = set(numeric_cols) | set(categorical_cols)
    assert "TARGET" not in all_cols
    assert "SK_ID_CURR" not in all_cols
    assert "CODE_GENDER" not in all_cols


def test_build_preprocessor_fits_and_transforms():
    df = generate_home_credit_synthetic(n_rows=300, seed=1)
    df = engineer_home_credit_features(df)
    numeric_cols, categorical_cols = split_feature_columns(
        df, target_col="TARGET", exclude_cols=["SK_ID_CURR", "CODE_GENDER", "AGE_GROUP"]
    )
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X = df[numeric_cols + categorical_cols]
    X_t = preprocessor.fit_transform(X)
    assert X_t.shape[0] == len(df)
    assert not np.isnan(X_t).any()
