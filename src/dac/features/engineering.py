"""Feature engineering for Home Credit and HMDA.

Deliberately generic column handling (auto-detect numeric vs. categorical)
so this also works unmodified against the real, much wider Kaggle
application_train.csv once scripts/download_home_credit.py has been run.
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

HOME_CREDIT_ANOMALY_SENTINEL = 365243


def engineer_home_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive human-interpretable features from Home Credit's DAYS_* / AMT_*
    columns. Mirrors the well-known feature-engineering conventions for this
    dataset (age in years, employment anomaly flag, income/credit ratios).
    """
    df = df.copy()

    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)

    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == HOME_CREDIT_ANOMALY_SENTINEL).astype(int)
    days_employed_clean = df["DAYS_EMPLOYED"].replace(HOME_CREDIT_ANOMALY_SENTINEL, np.nan)
    df["YEARS_EMPLOYED"] = (-days_employed_clean / 365.25).round(1)

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)
    df["DAYS_EMPLOYED_PERC"] = days_employed_clean / df["DAYS_BIRTH"].replace(0, np.nan)
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace(0, np.nan)
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCE_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    df["AGE_GROUP"] = pd.cut(
        df["AGE_YEARS"], bins=[0, 25, 35, 45, 55, 100], labels=["<25", "25-34", "35-44", "45-54", "55+"]
    ).astype(str)

    return df


def engineer_hmda_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LOAN_INCOME_RATIO"] = df["loan_amount"] / (df["income"] * 1000).replace(0, np.nan)
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
    """A standard impute+scale / impute+one-hot ColumnTransformer usable by
    both linear models (Logistic Regression) and tree ensembles.
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
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Flat list of feature names after ColumnTransformer.fit, for SHAP/plots."""
    return list(preprocessor.get_feature_names_out())
