"""Schema-accurate synthetic data generator for the UCI "default of credit
card clients" dataset (Yeh & Lien, 2009).

Stands in for the real UCI extract so the full EDA -> training -> tuning ->
XAI -> fairness pipeline can run end-to-end (and fast unit/integration tests
can run) without reading the full 30,000-row source file. Column names,
dtypes, and value domains mirror the real dataset's public codebook; only the
values are simulated.

IMPORTANT: a small, explicit synthetic bias term is injected into the
default probability as a function of SEX and AGE_GROUP. This is done ON
PURPOSE so that the fairness-audit and bias-mitigation stages of the
pipeline have a real, known effect to detect and correct -- it is not
present in, and should not be read as a claim about, the real-world data.
When real data is prepared (see scripts/prepare_uci_credit.py) this module
is only used as a fallback if that step hasn't been run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_uci_credit_synthetic(n_rows: int = 6_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic UCI credit-card rows", n_rows)

    ids = np.arange(1, n_rows + 1)
    limit_bal = rng.lognormal(mean=11.6, sigma=0.7, size=n_rows).round(-3).clip(10_000, 1_000_000)
    sex = rng.choice(["Male", "Female"], size=n_rows, p=[0.40, 0.60])
    education = rng.choice(
        ["Graduate School", "University", "High School", "Others", "Other/Unknown"],
        size=n_rows, p=[0.35, 0.47, 0.16, 0.01, 0.01],
    )
    marriage = rng.choice(["Married", "Single", "Others", "Other/Unknown"], size=n_rows, p=[0.45, 0.53, 0.015, 0.005])
    age = rng.integers(21, 80, size=n_rows)

    # PAY_0, PAY_2..PAY_6: repayment status codes (-2..8; <=0 roughly "paid
    # on time / no consumption", positive = months overdue). Simulate as a
    # per-customer latent repayment-risk tendency with month-to-month noise.
    latent_risk = rng.normal(0, 1, size=n_rows)
    pay_cols = {}
    for name in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        status = np.round(latent_risk * 1.3 + rng.normal(0, 1, size=n_rows) - 1.0).clip(-2, 8).astype(int)
        pay_cols[name] = status

    bill_base = limit_bal * rng.uniform(0.05, 0.75, size=n_rows)
    bill_cols, pay_amt_cols = {}, {}
    running_bill = bill_base
    for i in range(1, 7):
        running_bill = (running_bill * rng.uniform(0.85, 1.1, size=n_rows)).clip(0, None)
        bill_cols[f"BILL_AMT{i}"] = running_bill.round(2)
        pay_amt_cols[f"PAY_AMT{i}"] = (running_bill * rng.uniform(0.0, 0.4, size=n_rows)).round(2)

    age_group = pd.cut(
        age, bins=[0, 25, 35, 45, 55, 100], labels=["<25", "25-34", "35-44", "45-54", "55+"]
    ).astype(str)

    # NOTE: synthetic, deliberately-injected bias terms (see module docstring).
    sex_bias = np.where(sex == "Male", 0.10, -0.10)
    young_bias = np.where(np.isin(age_group, ["<25", "25-34"]), 0.12, -0.04)

    logit = (
        -1.4
        + 0.9 * latent_risk
        - 0.25 * np.log1p(limit_bal / 1e5)
        + 0.10 * np.log1p(bill_cols["BILL_AMT1"] / (limit_bal + 1))
        + sex_bias
        + young_bias
        + rng.normal(0, 0.5, size=n_rows)
    )
    default_prob = _sigmoid(logit)
    target = rng.binomial(1, default_prob)

    df = pd.DataFrame(
        {
            "ID": ids,
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            **pay_cols,
            **bill_cols,
            **pay_amt_cols,
            "DEFAULT_PAYMENT_NEXT_MONTH": target,
        }
    )

    logger.info("Synthetic UCI credit-card default rate: %.4f", df["DEFAULT_PAYMENT_NEXT_MONTH"].mean())
    return df
