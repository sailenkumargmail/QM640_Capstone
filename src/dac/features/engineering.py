"""Feature engineering for the UCI "default of credit card clients" dataset.

Deliberately generic column handling for the split-columns / preprocessor
helpers so the same code works against either the real prepared extract
(scripts/prepare_uci_credit.py) or the schema-accurate synthetic fallback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def engineer_uci_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive credit-utilization and repayment-behavior features from the
    six months of billing/payment history, mirroring standard credit-risk
    feature-engineering conventions for this dataset (Yeh & Lien, 2009).
    """
    df = df.copy()

    df["AGE_GROUP"] = pd.cut(
        df["AGE"], bins=[0, 25, 35, 45, 55, 100], labels=["<25", "25-34", "35-44", "45-54", "55+"]
    ).astype(str)

    df["AVG_BILL_AMT"] = df[BILL_AMT_COLS].mean(axis=1)
    df["AVG_PAY_AMT"] = df[PAY_AMT_COLS].mean(axis=1)
    df["BILL_LIMIT_RATIO"] = df["AVG_BILL_AMT"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["PAY_TO_BILL_RATIO"] = df["AVG_PAY_AMT"] / df["AVG_BILL_AMT"].replace(0, np.nan)
    df["BILL_AMT_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]

    df["MAX_DELINQUENCY"] = df[PAY_STATUS_COLS].max(axis=1)
    df["MEAN_DELINQUENCY"] = df[PAY_STATUS_COLS].mean(axis=1)
    df["MONTHS_DELINQUENT"] = (df[PAY_STATUS_COLS] > 0).sum(axis=1)

    return df


def split_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list[str],
) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for every column not in
    {target_col} | exclude_cols, based on dtype.
    """
    feature_cols = [c for c in df.columns if c not in exclude_cols and c != target_col]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """A standard impute+scale / impute+one-hot ColumnTransformer, shared
    identically by all four model families (Logistic Regression, XGBoost,
    CatBoost, EBM) so SHAP and fairness results are comparable across models.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    # Pandas output (not a bare ndarray) so downstream estimators -- notably
    # EBM, which reads column names directly off its input -- see the same
    # feature names SHAP is explicitly given via get_feature_names_out().
    preprocessor.set_output(transform="pandas")
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Flat list of feature names after ColumnTransformer.fit, for SHAP/plots."""
    return list(preprocessor.get_feature_names_out())
