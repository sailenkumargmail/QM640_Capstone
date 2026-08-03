import numpy as np

from dac.data.synthetic import generate_uci_credit_synthetic
from dac.features.engineering import (
    build_preprocessor,
    engineer_uci_credit_features,
    split_feature_columns,
)


def test_engineer_uci_credit_features_adds_columns():
    df = generate_uci_credit_synthetic(n_rows=300, seed=1)
    out = engineer_uci_credit_features(df)
    for col in ["AGE_GROUP", "AVG_BILL_AMT", "AVG_PAY_AMT", "BILL_LIMIT_RATIO", "MAX_DELINQUENCY", "MONTHS_DELINQUENT"]:
        assert col in out.columns


def test_age_group_buckets_cover_all_rows():
    df = generate_uci_credit_synthetic(n_rows=2000, seed=1)
    out = engineer_uci_credit_features(df)
    assert out["AGE_GROUP"].isin(["<25", "25-34", "35-44", "45-54", "55+"]).all()


def test_split_feature_columns_excludes_target_and_ids():
    df = generate_uci_credit_synthetic(n_rows=200, seed=1)
    df = engineer_uci_credit_features(df)
    numeric_cols, categorical_cols = split_feature_columns(
        df, target_col="DEFAULT_PAYMENT_NEXT_MONTH", exclude_cols=["ID", "SEX", "AGE_GROUP"]
    )
    all_cols = set(numeric_cols) | set(categorical_cols)
    assert "DEFAULT_PAYMENT_NEXT_MONTH" not in all_cols
    assert "ID" not in all_cols
    assert "SEX" not in all_cols


def test_build_preprocessor_fits_and_transforms():
    df = generate_uci_credit_synthetic(n_rows=300, seed=1)
    df = engineer_uci_credit_features(df)
    numeric_cols, categorical_cols = split_feature_columns(
        df, target_col="DEFAULT_PAYMENT_NEXT_MONTH", exclude_cols=["ID", "SEX", "AGE_GROUP"]
    )
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X = df[numeric_cols + categorical_cols]
    X_t = preprocessor.fit_transform(X)
    assert X_t.shape[0] == len(df)
    assert not np.isnan(X_t.to_numpy()).any()
