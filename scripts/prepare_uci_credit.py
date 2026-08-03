"""Prepares the real UCI "default of credit card clients" dataset for the
pipeline: reads the raw .xls export, applies light, documented cleanup, and
writes a plain CSV under data/raw/uci_credit/.

Source: Yeh, I-C., & Lien, C. (2009). The comparisons of data mining
techniques for the predictive accuracy of probability of default of credit
card clients. Expert Systems with Applications, 36(2), 2473-2480.
UCI ML Repository: https://archive.ics.uci.edu/dataset/350

Usage:
    python scripts/prepare_uci_credit.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SOURCE_XLS = REPO / "data" / "external" / "UCI" / "default of credit card clients.xls"
OUTPUT_CSV = REPO / "data" / "raw" / "uci_credit" / "default_of_credit_card_clients.csv"


def prepare() -> pd.DataFrame:
    df = pd.read_excel(SOURCE_XLS, header=1)

    df = df.rename(columns={"default payment next month": "DEFAULT_PAYMENT_NEXT_MONTH"})

    # SEX: 1 = male, 2 = female (UCI codebook) -> human-readable labels, kept
    # as a protected attribute for the fairness audit.
    df["SEX"] = df["SEX"].map({1: "Male", 2: "Female"})

    # EDUCATION codebook: 1=graduate school, 2=university, 3=high school,
    # 4=others; 0, 5, 6 are undocumented/unofficial codes present in the real
    # extract -- collapsed into "Other/Unknown" rather than dropped, so the
    # rows are retained but the anomaly is transparent.
    df["EDUCATION"] = df["EDUCATION"].map(
        {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
    ).fillna("Other/Unknown")

    # MARRIAGE codebook: 1=married, 2=single, 3=others; 0 is undocumented.
    df["MARRIAGE"] = df["MARRIAGE"].map({1: "Married", 2: "Single", 3: "Others"}).fillna("Other/Unknown")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols -> {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    prepare()
