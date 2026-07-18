"""Schema-accurate synthetic data generators.

These stand in for the real Kaggle "Home Credit Default Risk" and CFPB HMDA
datasets so the full EDA -> training -> tuning -> XAI -> fairness pipeline can
run end-to-end without Kaggle credentials. Column names, dtypes, and value
domains mirror the real datasets' public data dictionaries; only the values
are simulated.

IMPORTANT: a small, explicit synthetic bias term is injected into the
default/denial probability as a function of the protected attributes
(gender/age for Home Credit; race/sex for HMDA). This is done ON PURPOSE so
that the fairness-audit and bias-mitigation stages of the pipeline have a
real, known effect to detect and correct -- it is not present in, and should
not be read as a claim about, the real-world data. When real data is swapped
in (see scripts/download_home_credit.py and scripts/download_hmda.py) this
module is no longer used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_home_credit_synthetic(n_rows: int = 25_000, seed: int = 42) -> pd.DataFrame:
    """Simulate a subset of application_train.csv from the Home Credit Default
    Risk Kaggle competition (https://www.kaggle.com/c/home-credit-default-risk).
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic Home Credit rows", n_rows)

    sk_id_curr = np.arange(100_001, 100_001 + n_rows)

    code_gender = rng.choice(["F", "M"], size=n_rows, p=[0.66, 0.34])
    name_contract_type = rng.choice(
        ["Cash loans", "Revolving loans"], size=n_rows, p=[0.90, 0.10]
    )
    flag_own_car = rng.choice(["Y", "N"], size=n_rows, p=[0.34, 0.66])
    flag_own_realty = rng.choice(["Y", "N"], size=n_rows, p=[0.69, 0.31])
    cnt_children = rng.poisson(0.4, size=n_rows).clip(0, 10)

    amt_income_total = rng.lognormal(mean=11.9, sigma=0.45, size=n_rows).round(2)
    amt_credit = (amt_income_total * rng.uniform(1.5, 6.0, size=n_rows)).round(2)
    amt_annuity = (amt_credit / rng.uniform(8, 30, size=n_rows)).round(2)
    amt_goods_price = (amt_credit * rng.uniform(0.85, 1.0, size=n_rows)).round(2)

    name_type_suite = rng.choice(
        ["Unaccompanied", "Family", "Spouse, partner", "Children", "Other_A", "Other_B"],
        size=n_rows,
        p=[0.81, 0.13, 0.03, 0.015, 0.01, 0.005],
    )
    name_income_type = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Student", "Unemployed"],
        size=n_rows,
        p=[0.52, 0.23, 0.18, 0.06, 0.005, 0.005],
    )
    name_education_type = rng.choice(
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ],
        size=n_rows,
        p=[0.71, 0.24, 0.03, 0.017, 0.003],
    )
    name_family_status = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n_rows,
        p=[0.64, 0.15, 0.10, 0.06, 0.05],
    )
    name_housing_type = rng.choice(
        ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"],
        size=n_rows,
        p=[0.88, 0.048, 0.036, 0.02, 0.01, 0.006],
    )

    region_population_relative = rng.uniform(0.0005, 0.075, size=n_rows).round(6)

    # Ages 21-69, stored the Home Credit way: negative days relative to application date.
    age_years = rng.uniform(21, 69, size=n_rows)
    days_birth = (-age_years * 365.25).round().astype(int)

    # Employment: ~18% are pensioners/unemployed with the Home Credit sentinel 365243.
    days_employed = (-rng.uniform(0, 40, size=n_rows) * 365.25).round().astype(int)
    pensioner_mask = name_income_type == "Pensioner"
    days_employed = np.where(pensioner_mask, 365243, days_employed)

    days_registration = (-rng.uniform(0, 25, size=n_rows) * 365.25).round().astype(int)
    days_id_publish = (-rng.uniform(0, 12, size=n_rows) * 365.25).round().astype(int)
    own_car_age = np.where(
        flag_own_car == "Y", rng.uniform(0, 25, size=n_rows).round(1), np.nan
    )

    occupation_type = rng.choice(
        [
            "Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
            "High skill tech staff", "Accountants", "Medicine staff",
            "Security staff", "Cooking staff", "Cleaning staff", "Private service staff",
        ],
        size=n_rows,
    )
    cnt_fam_members = (cnt_children + rng.integers(1, 3, size=n_rows)).clip(1, 12).astype(float)

    region_rating_client = rng.choice([1, 2, 3], size=n_rows, p=[0.10, 0.70, 0.20])
    organization_type = rng.choice(
        ["Business Entity Type 3", "Self-employed", "School", "Government",
         "Trade: type 7", "Kindergarten", "Medicine", "XNA", "Construction", "Transport: type 4"],
        size=n_rows,
    )

    # EXT_SOURCE_1/2/3: the strongest real predictors of default; simulate as
    # informative external credit-bureau scores in [0, 1].
    ext_source_1 = rng.beta(5, 3, size=n_rows)
    ext_source_2 = rng.beta(5, 3, size=n_rows)
    ext_source_3 = rng.beta(5, 3, size=n_rows)

    obs_30_cnt_social_circle = rng.poisson(1.4, size=n_rows)
    def_30_cnt_social_circle = rng.binomial(obs_30_cnt_social_circle, 0.05)
    obs_60_cnt_social_circle = obs_30_cnt_social_circle
    def_60_cnt_social_circle = rng.binomial(obs_60_cnt_social_circle, 0.04)
    days_last_phone_change = (-rng.uniform(0, 10, size=n_rows) * 365.25).round().astype(int)

    amt_req_credit_bureau_year = rng.poisson(1.5, size=n_rows)

    # --- Target-generating process -----------------------------------
    # Higher external scores, income, age -> lower default risk.
    # Higher credit/income ratio, more social-circle defaults -> higher risk.
    credit_income_ratio = amt_credit / amt_income_total
    age_group = pd.cut(
        age_years,
        bins=[0, 25, 35, 45, 55, 100],
        labels=["<25", "25-34", "35-44", "45-54", "55+"],
    ).astype(str)

    # NOTE: synthetic, deliberately-injected bias terms (see module docstring).
    gender_bias = np.where(code_gender == "M", 0.12, -0.12)
    young_bias = np.where(np.isin(age_group, ["<25", "25-34"]), 0.10, -0.03)

    logit = (
        -1.8
        - 3.2 * (ext_source_1 - 0.5)
        - 3.0 * (ext_source_2 - 0.5)
        - 2.6 * (ext_source_3 - 0.5)
        + 0.35 * np.log1p(credit_income_ratio)
        - 0.15 * np.log1p(amt_income_total / 1e5)
        + 0.25 * def_30_cnt_social_circle
        - 0.01 * (age_years - 40)
        + gender_bias
        + young_bias
        + rng.normal(0, 0.55, size=n_rows)
    )
    default_prob = _sigmoid(logit)
    target = rng.binomial(1, default_prob)

    df = pd.DataFrame(
        {
            "SK_ID_CURR": sk_id_curr,
            "TARGET": target,
            "NAME_CONTRACT_TYPE": name_contract_type,
            "CODE_GENDER": code_gender,
            "FLAG_OWN_CAR": flag_own_car,
            "FLAG_OWN_REALTY": flag_own_realty,
            "CNT_CHILDREN": cnt_children,
            "AMT_INCOME_TOTAL": amt_income_total,
            "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity,
            "AMT_GOODS_PRICE": amt_goods_price,
            "NAME_TYPE_SUITE": name_type_suite,
            "NAME_INCOME_TYPE": name_income_type,
            "NAME_EDUCATION_TYPE": name_education_type,
            "NAME_FAMILY_STATUS": name_family_status,
            "NAME_HOUSING_TYPE": name_housing_type,
            "REGION_POPULATION_RELATIVE": region_population_relative,
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "DAYS_REGISTRATION": days_registration,
            "DAYS_ID_PUBLISH": days_id_publish,
            "OWN_CAR_AGE": own_car_age,
            "OCCUPATION_TYPE": occupation_type,
            "CNT_FAM_MEMBERS": cnt_fam_members,
            "REGION_RATING_CLIENT": region_rating_client,
            "ORGANIZATION_TYPE": organization_type,
            "EXT_SOURCE_1": ext_source_1,
            "EXT_SOURCE_2": ext_source_2,
            "EXT_SOURCE_3": ext_source_3,
            "OBS_30_CNT_SOCIAL_CIRCLE": obs_30_cnt_social_circle,
            "DEF_30_CNT_SOCIAL_CIRCLE": def_30_cnt_social_circle,
            "OBS_60_CNT_SOCIAL_CIRCLE": obs_60_cnt_social_circle,
            "DEF_60_CNT_SOCIAL_CIRCLE": def_60_cnt_social_circle,
            "DAYS_LAST_PHONE_CHANGE": days_last_phone_change,
            "AMT_REQ_CREDIT_BUREAU_YEAR": amt_req_credit_bureau_year,
            "AGE_GROUP": age_group,
        }
    )

    # Inject realistic missingness (Home Credit has heavy missingness in several columns).
    for col, frac in [
        ("EXT_SOURCE_1", 0.28),
        ("OWN_CAR_AGE", 0.66),
        ("OCCUPATION_TYPE", 0.05),
        ("AMT_ANNUITY", 0.001),
        ("AMT_GOODS_PRICE", 0.001),
        ("CNT_FAM_MEMBERS", 0.0005),
    ]:
        mask = rng.random(n_rows) < frac
        df.loc[mask, col] = np.nan

    logger.info("Synthetic Home Credit default rate: %.4f", df["TARGET"].mean())
    return df


def generate_hmda_synthetic(n_rows: int = 15_000, seed: int = 7) -> pd.DataFrame:
    """Simulate a subset of a legacy (pre-2018) HMDA Loan/Application Register
    (LAR) extract of ORIGINATED loans (https://ffiec.cfpb.gov/data-browser/),
    matching the schema produced by scripts/prepare_hmda_real.py from the real
    data/external/HMDA/ files. Used ONLY for fairness auditing of a pricing
    model -- HMDA here records already-originated loans and a higher-priced
    (high_cost_flag) outcome, not loan performance, so it is never treated as
    a "default" target (see Synopsis_Credit_Default_XAI_Bias_Mitigation.docx,
    Background and Context, and the reviewer critique in response.txt,
    Cross-Question 1).
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic HMDA rows", n_rows)

    loan_id = [f"HMDA-{i:07d}" for i in range(1, n_rows + 1)]

    derived_race = rng.choice(
        ["White", "Black or African American", "Asian", "American Indian or Alaska Native",
         "Native Hawaiian or Other Pacific Islander", "2 or more minority races",
         "Joint", "Race Not Available"],
        size=n_rows,
        p=[0.62, 0.10, 0.08, 0.01, 0.005, 0.015, 0.06, 0.11],
    )
    derived_ethnicity = rng.choice(
        ["Not Hispanic or Latino", "Hispanic or Latino", "Joint", "Ethnicity Not Available"],
        size=n_rows,
        p=[0.73, 0.11, 0.05, 0.11],
    )
    derived_sex = rng.choice(["Male", "Female", "Joint", "Sex Not Available"], size=n_rows, p=[0.44, 0.33, 0.15, 0.08])

    loan_type = rng.choice(["Conventional", "FHA", "VA", "USDA"], size=n_rows, p=[0.72, 0.17, 0.08, 0.03])
    loan_purpose = rng.choice(
        ["Home purchase", "Refinancing", "Cash-out refinancing", "Home improvement"],
        size=n_rows, p=[0.45, 0.25, 0.20, 0.10],
    )
    occupancy_type = rng.choice(
        ["Principal residence", "Second residence", "Investment property"],
        size=n_rows, p=[0.86, 0.06, 0.08],
    )

    income = rng.lognormal(mean=11.2, sigma=0.55, size=n_rows).round(0)  # thousands USD, HMDA convention
    loan_amount = (income * rng.uniform(1.5, 4.5, size=n_rows) * 1000).round(-3)
    property_value = (loan_amount * rng.uniform(1.05, 1.6, size=n_rows)).round(-3)
    loan_to_value_ratio = (loan_amount / property_value * 100).round(2)
    debt_to_income_ratio = rng.choice(
        ["<20%", "20%-<30%", "30%-<36%", "36%-<43%", "43%-<50%", "50%-60%", ">60%"],
        size=n_rows,
        p=[0.12, 0.18, 0.20, 0.24, 0.14, 0.08, 0.04],
    )
    interest_rate = rng.normal(6.5, 1.1, size=n_rows).clip(2.5, 12).round(3)
    applicant_age = rng.choice(
        ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"],
        size=n_rows,
        p=[0.05, 0.22, 0.24, 0.20, 0.15, 0.09, 0.05],
    )
    state_code = rng.choice(["MI", "OH", "IN", "IL", "WI"], size=n_rows)

    dti_risk = pd.Series(debt_to_income_ratio).map(
        {"<20%": -0.6, "20%-<30%": -0.3, "30%-<36%": -0.1, "36%-<43%": 0.15,
         "43%-<50%": 0.5, "50%-60%": 0.9, ">60%": 1.3}
    ).to_numpy()

    # NOTE: synthetic, deliberately-injected bias terms (see module docstring).
    # Higher-cost-loan risk rises with rate, DTI, LTV; falls with income.
    race_bias = np.where(
        np.isin(derived_race, ["Black or African American", "American Indian or Alaska Native"]),
        0.35,
        0.0,
    )

    logit = (
        -2.6
        + 0.35 * (interest_rate - 6.5)
        + 0.9 * dti_risk
        + 0.02 * (loan_to_value_ratio - 80)
        - 0.20 * np.log1p(income / 60)
        + race_bias
        + rng.normal(0, 0.5, size=n_rows)
    )
    high_cost_prob = _sigmoid(logit)
    high_cost_flag = rng.binomial(1, high_cost_prob)  # 1 = higher-priced/HOEPA-reportable loan

    df = pd.DataFrame(
        {
            "loan_id": loan_id,
            "high_cost_flag": high_cost_flag,
            "derived_race": derived_race,
            "derived_ethnicity": derived_ethnicity,
            "derived_sex": derived_sex,
            "loan_type": loan_type,
            "loan_purpose": loan_purpose,
            "occupancy_type": occupancy_type,
            "income": income,
            "loan_amount": loan_amount,
            "property_value": property_value,
            "loan_to_value_ratio": loan_to_value_ratio,
            "debt_to_income_ratio": debt_to_income_ratio,
            "interest_rate": interest_rate,
            "applicant_age": applicant_age,
            "state_code": state_code,
        }
    )

    for col, frac in [("interest_rate", 0.03), ("property_value", 0.02), ("income", 0.01)]:
        mask = rng.random(n_rows) < frac
        df.loc[mask, col] = np.nan

    logger.info("Synthetic HMDA high-cost-loan rate: %.4f", df["high_cost_flag"].mean())
    return df
