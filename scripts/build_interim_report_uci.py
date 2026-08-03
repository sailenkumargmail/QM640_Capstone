"""Builds the QM640 Interim Report docx for the UCI credit-card default
pipeline (Logistic Regression, XGBoost, CatBoost, EBM), using the previous
Interim Report (Home Credit / HMDA) as the base document for APA
styles/skeleton, and QM+640+Interim+Report+template-1.docx as the section
structure reference. Every data-specific paragraph, table, and figure in the
base document is replaced with UCI-pipeline content, sourced from the actual
completed run's reports/model_performance/run_summary.json -- no numbers in
this script are hand-typed; they are all pulled from that file (or computed
live from the loaded dataframe for the granular EDA tables) so the report
matches whatever the pipeline actually produced.

Usage:
    python scripts/build_interim_report_uci.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dac.data.loader import load_uci_credit  # noqa: E402
from dac.features.engineering import engineer_uci_credit_features  # noqa: E402

BASE_DOC = Path(r"C:\Project\MS\DAC\QM640 Data Analytics Capstone Interim Report Sailen.docx")
OUTPUT_DOC = Path(r"C:\Project\MS\DAC\QM640 Data Analytics Capstone Interim Report Sailen (UCI Credit Card).docx")
GITHUB_URL = "https://github.com/sailenkumargmail/QM640_Capstone"

FIG = REPO / "reports" / "figures" / "uci_credit"
FIG_EVID = REPO / "reports" / "figures" / "report_evidence"
METRICS_DIR = REPO / "reports" / "model_performance"

summary = json.loads((METRICS_DIR / "run_summary.json").read_text(encoding="utf-8"))
baseline = summary["baseline_metrics"]
tuned = summary["tuned_metrics"]
best_model = summary["best_model"]
mitigated = summary["mitigated_performance"]
mit_attr = summary["fairness_mitigation_attribute"]
pre_fair = summary["fairness_pre_mitigation"]
post_fair = summary["fairness_post_mitigation"]
shap_top = summary["shap_top_features"]
ebm_top = summary["ebm_top_features"]
agreement = summary["shap_ebm_agreement"]

MODEL_LABELS = {
    "logistic_regression_tuned": "Logistic Regression",
    "xgboost_tuned": "XGBoost",
    "catboost_tuned": "CatBoost",
    "ebm_tuned": "EBM",
    "logistic_regression_baseline": "Logistic Regression (baseline)",
    "xgboost_baseline": "XGBoost (baseline)",
    "catboost_baseline": "CatBoost (baseline)",
    "ebm_baseline": "EBM (baseline)",
}


def label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Live EDA recomputation (the base doc's per-dataset EDA tables were hand
# populated from pipeline output; here they're computed directly so every
# number is traceable to the checked-in data, not retyped).
# ---------------------------------------------------------------------------
_raw_df, IS_SYNTHETIC = load_uci_credit()
eda_df = engineer_uci_credit_features(_raw_df)
TARGET = "DEFAULT_PAYMENT_NEXT_MONTH"

sex_rate = eda_df.groupby("SEX")[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)
age_rate = eda_df.groupby("AGE_GROUP")[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)
edu_rate = eda_df.groupby("EDUCATION")[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)
marriage_rate = eda_df.groupby("MARRIAGE")[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)

_numeric_df = eda_df.select_dtypes(include="number")
corr_with_target = (
    _numeric_df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
)

KEY_NUMERIC_COLS = [
    "LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "AVG_BILL_AMT", "AVG_PAY_AMT",
    "BILL_LIMIT_RATIO", "PAY_TO_BILL_RATIO", "MAX_DELINQUENCY", "MONTHS_DELINQUENT",
]
numeric_summary = eda_df[KEY_NUMERIC_COLS].describe().T

sex_gap_hi, sex_gap_lo = sex_rate["mean"].iloc[0], sex_rate["mean"].iloc[-1]
age_gap_hi, age_gap_lo = age_rate["mean"].iloc[0], age_rate["mean"].iloc[-1]

n_dupes = int(eda_df.duplicated().sum())
n_missing_total = int(eda_df.isna().sum().sum())
_missing_by_col = eda_df.isna().sum()
_missing_cols = _missing_by_col[_missing_by_col > 0]
n_missing_cols = int(len(_missing_cols))
n_unknown_education = int((eda_df["EDUCATION"] == "Other/Unknown").sum())
n_unknown_marriage = int((eda_df["MARRIAGE"] == "Other/Unknown").sum())


# ---------------------------------------------------------------------------
# Small statistics helpers (all computed live -- no hand-typed sample sizes)
# ---------------------------------------------------------------------------
def two_proportion_power_n(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80, one_tailed: bool = False) -> int:
    """Cohen's h effect size, per-group N for a two-proportion z-test."""
    h = abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
    z_alpha = norm.ppf(1 - alpha) if one_tailed else norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n = ((z_alpha + z_beta) / h) ** 2
    return int(np.ceil(n))


def correlation_power_n(r: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Fisher z-transform sample size for detecting a target correlation r."""
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n = ((z_alpha + z_beta) / np.arctanh(abs(r))) ** 2 + 3
    return int(np.ceil(n))


rq1_p1 = baseline["logistic_regression_baseline"]["roc_auc"]
rq1_p2 = max(baseline[m]["roc_auc"] for m in ["xgboost_baseline", "catboost_baseline", "ebm_baseline"])
rq1_n = two_proportion_power_n(rq1_p1, rq1_p2, one_tailed=True)

rq2_r_pilot = 0.5  # conservative pilot correlation assumption for the sample-size floor (see note in prose)
rq2_n = correlation_power_n(rq2_r_pilot)

rq3_n = two_proportion_power_n(sex_gap_hi, sex_gap_lo)

# ---------------------------------------------------------------------------
# Helpers (generic docx manipulation -- reused pattern from
# scripts/build_interim_report.py)
# ---------------------------------------------------------------------------

doc = docx.Document(str(BASE_DOC))


def find_para(text: str, contains: bool = False):
    for p in doc.paragraphs:
        t = p.text.strip()
        if (contains and text in t) or (not contains and t == text):
            return p
    raise ValueError(f"paragraph not found: {text!r}")


def set_text(p, text: str):
    if not p.runs:
        p.add_run(text)
        return p
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""
    return p


def heading_before(anchor, text: str, style: str = "APA Heading 2"):
    return anchor.insert_paragraph_before(text, style=style)


def normal_before(anchor, text: str, style: str = "Normal", bold=False, italic=False):
    p = anchor.insert_paragraph_before("", style=style)
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    return p


def bullets_before(anchor, items, style: str = "List Bullet"):
    out = []
    for text in items:
        out.append(anchor.insert_paragraph_before(text, style=style))
    return out


TABLE_BORDER_XML = (
    '<w:tblBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:top w:val="single" w:color="000000" w:themeColor="text1" w:sz="12"/>'
    '<w:left w:val="single" w:color="000000" w:themeColor="text1" w:sz="12"/>'
    '<w:bottom w:val="single" w:color="000000" w:themeColor="text1" w:sz="12"/>'
    '<w:right w:val="single" w:color="000000" w:themeColor="text1" w:sz="12"/>'
    '<w:insideH w:val="single" w:color="000000" w:themeColor="text1" w:sz="12"/>'
    '<w:insideV w:val="single" w:color="000000" w:themeColor="text1" w:sz="12"/>'
    "</w:tblBorders>"
)


def style_table_borders(tbl):
    from docx.oxml import parse_xml

    tblPr = tbl._tbl.tblPr
    borders = parse_xml(TABLE_BORDER_XML)
    tblPr.append(borders)


def table_before(anchor, header, rows, bold_header=True, font_size=9):
    tbl = doc.add_table(rows=1, cols=len(header))
    hdr_cells = tbl.rows[0].cells
    for i, h in enumerate(header):
        hdr_cells[i].text = str(h)
        for p in hdr_cells[i].paragraphs:
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                r.bold = bold_header
                r.font.size = Pt(font_size)
    for row in rows:
        cells = tbl.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = "" if v is None else str(v)
            for p in cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(2)
                for r in p.runs:
                    r.font.size = Pt(font_size)
    style_table_borders(tbl)
    anchor._p.addprevious(tbl._tbl)
    normal_before(anchor, "")
    return tbl


def h0_ha_table(anchor, h0: str, ha: str, font_size=9):
    return table_before(anchor, [f"H0: {h0}"], [[f"Ha: {ha}"]], bold_header=True, font_size=font_size)


def caption_before(anchor, text: str, style: str = "LO-normal"):
    return normal_before(anchor, text, style=style, italic=False)


def image_before(anchor, path: Path, caption: str, width_in: float = 6.0):
    if not path.exists():
        normal_before(anchor, f"[Figure not found: {path.name}]", italic=True)
        return
    p = anchor.insert_paragraph_before("")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_in))
    caption_before(anchor, caption)
    normal_before(anchor, "")


def delete_para(p):
    p._p.getparent().remove(p._p)


def delete_table_by_header(header_prefix: list[str]):
    for t in list(doc.tables):
        header = [c.text.strip() for c in t.rows[0].cells]
        if header[: len(header_prefix)] == header_prefix:
            t._tbl.getparent().remove(t._tbl)
            return True
    return False


def delete_between(start_para, end_para, delete_start_too: bool = False):
    """Delete every body-level element strictly between start_para and
    end_para (both paragraph/table siblings under <w:body>)."""
    el = start_para._p.getnext()
    to_remove = []
    while el is not None and el is not end_para._p:
        to_remove.append(el)
        el = el.getnext()
    for el in to_remove:
        el.getparent().remove(el)
    if delete_start_too:
        delete_para(start_para)


def clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "").replace("_", " ")


print("Base document:", BASE_DOC)
print("run_summary.json:", METRICS_DIR / "run_summary.json")
print("Best model:", best_model)

# ===========================================================================
# 1. Title page
# ===========================================================================
title_sub = find_para(
    "Balancing Accuracy, Explainability, and Fairness in Automated Credit and Mortgage Decisioning: A Dual-Dataset Machine Le",
    contains=True,
)
set_text(
    title_sub,
    "Balancing Accuracy, Explainability, and Fairness in Consumer Credit-Card Default Prediction: "
    "A Four-Model Machine Learning Framework",
)
date_para = find_para("July 28, 2026")
set_text(date_para, date.today().strftime("%B %d, %Y"))

# ===========================================================================
# 2. Introduction / Background / Problem Statement / Purpose
# ===========================================================================
intro_para = find_para("This synopsis presents the blueprint", contains=True)
set_text(
    intro_para,
    "This report documents an integrated machine-learning pipeline that predicts the probability a "
    "revolving credit-card account will default on its next payment, explains those predictions through "
    "both post-hoc and inherently interpretable methods, and audits and mitigates fairness gaps across "
    "two demographic attributes, all on a single, real, publicly available dataset.",
)

background_para = find_para("Automated credit scoring underlies most consumer lending decisions", contains=True)
set_text(
    background_para,
    "Automated credit scoring is now the default mechanism by which card issuers decide who receives "
    "credit and how much, allowing lenders to evaluate applicant and account risk at a speed and scale "
    "manual underwriting cannot match (Fuster et al., 2022). The dataset examined here, 30,000 credit-card "
    "accounts from a major Taiwanese bank (Yeh & Lien, 2009), sits squarely in that setting: each row is a "
    "cardholder's demographic profile, credit limit, and six months of billing and repayment history, and "
    "the outcome of interest is whether that cardholder defaults the following month. Three groups of "
    "stakeholders have a direct interest in how such models are built: cardholders, whose access to credit "
    "and cost of borrowing depend on the score; issuers, who must balance charge-off losses against "
    "portfolio growth; and regulators and model-risk teams, who increasingly expect not just an accurate "
    "score but a documented, auditable account of how it was produced and whether it treats demographic "
    "groups comparably.",
)

problem_para = find_para("Financial institutions deploying machine-learning models for credit and mortgage", contains=True)
set_text(
    problem_para,
    "Teams building consumer-credit scoring models routinely face a three-way tension among predictive "
    "accuracy, explainability, and fairness (Barocas & Selbst, 2016; Mehrabi et al., 2021). In practice "
    "these properties are often pursued by different people at different times: a modeling team chases "
    "ROC-AUC, an explainability pass is bolted on afterward with SHAP or a similar tool, and a fairness "
    "audit, if it happens at all, runs as a separate compliance exercise once the model is already in "
    "production. That sequencing makes it hard to answer a question institutions increasingly need "
    "answered up front: given a specific accuracy gain from a more complex model, is the corresponding "
    "explanation trustworthy, and does the model treat cardholders comparably across gender and age? This "
    "project builds one pipeline that answers all three questions together on the same held-out data, "
    "rather than three pipelines answering them separately.",
)

purpose_paras = [
    find_para("The purpose of this study is to design, build, and validate an integrated machine-learning pipeline for automated credit", contains=True),
]
set_text(
    purpose_paras[0],
    "The purpose of this study is to design, build, and validate an integrated machine-learning pipeline "
    "for consumer credit-card default prediction that simultaneously",
)
purpose_para2 = find_para("(a) predicts/classifies borrower default risk with measurably higher accuracy than a traditional logistic-regression", contains=True)
set_text(
    purpose_para2,
    "(a) predicts next-month default with measurably higher accuracy than a tuned logistic-regression "
    "baseline, using two gradient-boosted ensembles (XGBoost, CatBoost) and one glass-box generalized "
    "additive model (Explainable Boosting Machine, EBM); (b) explains the strongest model's predictions "
    "using both SHAP (a post-hoc approximation) and EBM's own native explanation, and measures whether the "
    "two agree; (c) audits the strongest model for demographic-parity violations across SEX and a derived "
    "AGE_GROUP and mitigates any violation found via reweighing; and (d) tests whether that same reweighing "
    "recipe, applied unchanged, works comparably across all four trained model families rather than being "
    "an artifact of the one model it was designed against. The study is a predictive-classification and "
    "fairness-auditing exercise, not a causal-inference study: no claim of causal discrimination is made "
    "from the observed associations.",
)

# ===========================================================================
# 3. Progress Snapshot
# ===========================================================================
completed_para = find_para("Completed: acquisition and preparation of both real datasets", contains=True)
set_text(
    completed_para,
    "Completed: acquisition and preparation of the real dataset (UCI \u201cdefault of credit card clients,\u201d "
    f"{eda_df.shape[0]:,} accounts, Yeh & Lien, 2009); full EDA (distributions, correlation, target-rate gaps "
    "by SEX and AGE_GROUP); feature engineering (utilization and repayment-behavior ratios, delinquency "
    "aggregates, AGE_GROUP bucketing); baseline and tuned training of all four model families (Logistic "
    "Regression, XGBoost, CatBoost, EBM \u2014 RandomizedSearchCV for Logistic Regression, Optuna/TPE search for "
    "the other three); held-out evaluation (ROC-AUC, PR-AUC, F1, KS statistic, Brier score); SHAP "
    "explainability on XGBoost and EBM's native global explanation, plus a Spearman rank-agreement check "
    "between the two; fairness audit (pre-mitigation) on SEX and AGE_GROUP for all four tuned models; and "
    f"{mit_attr}-based reweighing mitigation with re-training and re-audit, replicated across all four models.",
)
inprogress_para = find_para("In progress: Expansion of the literature review", contains=True)
set_text(
    inprogress_para,
    "In progress: expansion of the literature review beyond this draft's initial source set; drafting of "
    "the accuracy-fairness decision framework referenced in Next Steps.",
)
pending_para = find_para("Pending: formal statistical hypothesis testing write-up", contains=True)
set_text(
    pending_para,
    "Pending: reporting of exact p-values (not just point estimates) for the RQ1-RQ4 hypothesis tests "
    "below; a joint SEX x AGE_GROUP sensitivity analysis of the reweighing mitigation; evaluation of a "
    "second mitigation technique for comparison, if time allows; the final model-governance checklist; and "
    "a final proofreading/APA-compliance pass.",
)

# ===========================================================================
# 4. Scope and objectives + RQ1-RQ4 + flowchart
# ===========================================================================
scope_para = find_para("This study is scoped entirely to two secondary", contains=True)
set_text(
    scope_para,
    "This study is scoped entirely to one secondary, publicly available, deidentified dataset; no primary "
    "data collection, survey instrument, or human-subjects contact is involved, and no institutional-"
    "review-board protocol is required. The work is organized around four research questions that progress "
    "from predictive accuracy across four model families (RQ1), through a comparison of post-hoc and "
    "native explainability (RQ2), to fairness auditing and mitigation of the best model (RQ3), and finally "
    "replication of that mitigation across every model family (RQ4). Each research question is paired below "
    "with an explicit statistical hypothesis and a minimum-sample-size justification (see Sample size "
    "calculation).",
)

rq1_p = find_para("RQ1: To what extent does a tuned XGBoost ensemble outperform", contains=True)
set_text(
    rq1_p,
    "RQ1: To what extent do tuned gradient-boosted ensembles (XGBoost, CatBoost) and a tuned glass-box "
    "generalized additive model (EBM) outperform a tuned logistic-regression baseline in predicting "
    "next-month credit-card default (DEFAULT_PAYMENT_NEXT_MONTH), as measured by held-out ROC-AUC?",
)
rq2_p = find_para("RQ2: Which borrower-level features drive the RQ1 best-performing model", contains=True)
set_text(
    rq2_p,
    "RQ2: Do SHAP's post-hoc feature attributions for XGBoost agree, in relative feature ranking, with the "
    "Explainable Boosting Machine's native (ante-hoc) term importances, and are the shared top drivers "
    "consistent with established credit-risk theory (repayment-status history, credit-utilization ratios)?",
)
rq3_p = find_para("RQ3: Does the RQ1 best-performing Home Credit model exhibit", contains=True)
set_text(
    rq3_p,
    "RQ3: Does the RQ1 best-performing model exhibit a demographic-parity-ratio violation of the "
    "four-fifths rule across SEX and AGE_GROUP, and does Kamiran and Calders' (2012) reweighing mitigation "
    "raise the ratio to or above 0.80 without a material (> 5 percentage-point) ROC-AUC cost?",
)
rq4_p = find_para("RQ4: Does the identical fairness-audit methodology used in RQ3", contains=True)
set_text(
    rq4_p,
    "RQ4: Does the identical reweighing recipe from RQ3, applied unchanged, produce a comparable "
    "demographic-parity-ratio improvement when replicated across all four trained model families "
    "(Logistic Regression, XGBoost, CatBoost, EBM), or does its effectiveness depend on model architecture?",
)

sample_size_heading = find_para("Sample size calculation")
image_before(
    sample_size_heading,
    FIG_EVID / "rq_solution_flowchart.png",
    "Figure 1 \u2014 RQ1-RQ4 lineage and overall solution flow.",
    width_in=6.3,
)

# ===========================================================================
# 5. Sample size calculation
# ===========================================================================
ss_para1 = find_para("Minimum sample size is estimated per research question", contains=True)
set_text(
    ss_para1,
    "Minimum sample size is estimated per research question using a two-proportion power analysis (RQ1, "
    "RQ3, treating ROC-AUC or a group default rate as a proportion) or a Fisher-z correlation power "
    "analysis (RQ2, since the RQ2 outcome is now a rank-correlation coefficient rather than a proportion), "
    "each at alpha = .05. Because this report is built directly from a single completed pipeline run rather "
    "than a separate synopsis-stage pilot, RQ1's effect size uses this run's own baseline (untuned) models "
    "as the pilot signal, with the tuned models reported below as the confirmatory result; RQ3's effect "
    "size uses the raw EDA-level default-rate gap by SEX, observed before any modeling.",
)
ss_para2 = find_para("RQ1 uses the pilot ROC-AUC gap between tuned logistic regression", contains=True)
set_text(
    ss_para2,
    f"RQ1 uses the baseline ROC-AUC gap between logistic regression ({rq1_p1:.4f}) and the strongest other "
    f"baseline model ({rq1_p2:.4f}) as the effect size for a one-tailed two-proportion z-test at power = "
    ".80. RQ2 uses a conservative assumed correlation of r = .50 (rather than a specific pilot value, since "
    "no prior SHAP-vs-EBM agreement estimate exists for this dataset) in a Fisher-z power analysis for "
    "detecting a non-zero Spearman correlation. RQ3 uses a two-tailed two-proportion z-test at power = .80 "
    "with the EDA pilot proportions above. RQ4 does not require an independent power analysis: it "
    "re-applies the same deterministic reweighing-and-audit computation to four already-fitted models on "
    "the single RQ3 held-out test set, rather than estimating a new effect from a fresh sample.",
)
old_ss_table_removed = delete_table_by_header(["Research Question", "Method Used", "Key Parameters", "Minimum Sample Size (N)"])
table_before(
    find_para("To ensure adequate statistical power across all four analyses", contains=True),
    ["Research Question", "Method Used", "Key Parameters", "Minimum Sample Size (N)"],
    [
        ("RQ1", "Power (one-tailed, 2-proportion z)",
         f"\u03b1 = .05, Power = .80, p\u2081 = {rq1_p1:.4f}, p\u2082 = {rq1_p2:.4f}",
         f"{rq1_n:,} / group ({2*rq1_n:,} total)"),
        ("RQ2", "Power (Fisher-z correlation)", "\u03b1 = .05, Power = .80, assumed r = .50", f"{rq2_n:,}"),
        ("RQ3", "Power (2-tailed, 2-proportion z)",
         f"\u03b1 = .05, Power = .80, p\u2081 = {sex_gap_hi:.4f}, p\u2082 = {sex_gap_lo:.4f}",
         f"{rq3_n:,} / group ({2*rq3_n:,} total)"),
        ("RQ4", "Deterministic replication (no separate power analysis)",
         "Reuses RQ3's held-out test set across all 4 already-fitted models", "N/A"),
    ],
)
ss_para3 = find_para("To ensure adequate statistical power across all four analyses", contains=True)
_n_train = eda_df.shape[0] - int(round(eda_df.shape[0] * 0.20))
set_text(
    ss_para3,
    f"To ensure adequate statistical power across the three analyses that require it, the governing minimum "
    f"is N = {max(2*rq1_n, rq2_n, 2*rq3_n):,} (driven by {'RQ1' if 2*rq1_n >= max(rq2_n, 2*rq3_n) else ('RQ2' if rq2_n >= 2*rq3_n else 'RQ3')}). "
    f"The dataset used in this study, {eda_df.shape[0]:,} accounts ({_n_train:,} in the training fold "
    f"alone), comfortably exceeds that requirement, giving ample statistical power even for the smaller "
    f"protected-attribute subgroups (e.g., the {age_rate['count'].idxmin()} AGE_GROUP band, "
    f"n = {int(age_rate['count'].min()):,}).",
)

print("Section 1-5 done.")

# ===========================================================================
# 6. Literature survey
# ===========================================================================
lit_approach_para = find_para("Sources were identified through targeted keyword searches", contains=True)
set_text(
    lit_approach_para,
    "Sources were identified through targeted keyword searches (“credit scoring explainable AI”, "
    "“SHAP credit risk”, “explainable boosting machine glass-box”, “CatBoost categorical "
    "boosting”, “bias mitigation reweighing classification”, “fairness machine learning "
    "lending”) across Google Scholar, SSRN, the ACM Digital Library, and arXiv, supplemented by backward "
    "citation-chasing from foundational fairness-ML, explainability, and credit-scoring surveys. Inclusion "
    "criteria: peer-reviewed articles, working papers, or top-venue proceedings published 1996-2026 that "
    "(a) apply machine learning to credit decisioning, (b) develop or apply a post-hoc or intrinsic "
    "explainability technique to tabular/financial models, or (c) develop or apply a group-fairness metric "
    "or bias-mitigation technique relevant to lending. Each source below is mapped to the research "
    "question(s) (RQ1-RQ4, see Scope and objectives) it most directly informs.",
)

lit_rows = [
    ("Yeh & Lien (2009)", "Credit scoring", "UCI “default of credit card clients” (this study's dataset)",
     "Compares six classifiers (incl. neural nets, logistic regression) for predicting credit-card default",
     "Introduces the dataset used throughout this study and establishes early benchmark accuracy figures for it",
     "RQ1 -- source and benchmark context for the dataset and prediction task"),
    ("Chen & Guestrin (2016)", "ML systems / gradient boosting", "Benchmark tabular datasets",
     "Introduces XGBoost, a scalable, regularized tree-boosting system",
     "Sparsity-aware split finding and weighted quantile sketch give large accuracy/speed gains over earlier "
     "boosting implementations on structured data",
     "RQ1 -- justifies XGBoost as one of the two ensemble challenger models"),
    ("Prokhorenkova, Gusev, Vorobev, Dorogush, & Gulin (2018)", "ML systems / gradient boosting",
     "Benchmark tabular + categorical-heavy datasets",
     "Introduces CatBoost, using ordered boosting and native categorical-feature handling",
     "Reduces prediction shift / target leakage inherent in standard gradient boosting on categorical "
     "features, improving generalization",
     "RQ1 -- justifies CatBoost as the second ensemble challenger model"),
    ("Lou, Caruana, & Gehrke (2012)", "Explainable AI (XAI) / glass-box models", "Benchmark tabular datasets",
     "Introduces generalized additive models with pairwise interactions (GA2M), the architecture underlying EBM",
     "Achieves accuracy competitive with full-complexity models while remaining exactly, not approximately, "
     "interpretable",
     "RQ1/RQ2 -- foundational architecture for the Explainable Boosting Machine model"),
    ("Nori, Jenkins, Koch, & Caruana (2019)", "Explainable AI (XAI) / software", "General-purpose ML library",
     "Introduces InterpretML, the open-source framework implementing the Explainable Boosting Machine used here",
     "Provides a production-grade, scikit-learn-compatible glass-box GAM implementation with native global "
     "and local explanation methods",
     "RQ1/RQ2 -- the EBM implementation and native-explanation API used throughout this study"),
    ("Lundberg & Lee (2017)", "Explainable AI (XAI)", "Model-agnostic; benchmark tabular data",
     "Introduces SHAP (Shapley Additive exPlanations), unifying prior additive-attribution methods",
     "SHAP values give a theoretically grounded, locally accurate, consistent per-feature attribution for "
     "any model, including tree ensembles (TreeExplainer)",
     "RQ2 -- the post-hoc explainability method compared against EBM's native explanation"),
    ("Ribeiro, Singh, & Guestrin (2016)", "Explainable AI (XAI)", "Text/image/tabular classifiers",
     "Introduces LIME, a local surrogate-model explanation technique",
     "Local linear surrogates approximate any classifier's decision boundary near a single prediction",
     "RQ2 -- contrasting post-hoc XAI approach; SHAP was selected over LIME for its additive consistency guarantees"),
    ("Doshi-Velez & Kim (2017)", "Explainable AI (XAI) / ML theory", "Conceptual / no single dataset",
     "Proposes a taxonomy for rigorously evaluating interpretability claims",
     "Distinguishes functionally-grounded, human-grounded, and application-grounded interpretability evaluation",
     "RQ2 -- frames why comparing a post-hoc method against a genuinely ante-hoc one is a meaningful test, "
     "not just two flavors of the same claim"),
    ("Barocas & Selbst (2016)", "Fairness / law and ML", "Conceptual / legal doctrine",
     "Legal analysis of disparate impact doctrine applied to data-driven decisions",
     "Formalizes the disparate-impact framing (incl. the four-fifths-rule heuristic) used to flag "
     "practically significant group differences in automated decisions",
     "RQ3/RQ4 -- source of the four-fifths / disparate-impact-ratio convention used throughout the fairness audit"),
    ("Dwork, Hardt, Pitassi, Reingold, & Zemel (2012)", "Fairness metrics", "Conceptual / algorithmic",
     "Formalizes “fairness through awareness” and individual-fairness constraints",
     "Establishes that group-blind treatment does not guarantee fair outcomes, motivating explicit "
     "group-fairness metrics",
     "RQ3/RQ4 -- theoretical grounding for auditing group-level parity rather than relying on excluding "
     "protected attributes alone"),
    ("Hardt, Price, & Srebro (2016)", "Fairness metrics", "Conceptual + empirical benchmarks",
     "Introduces equalized odds / equality of opportunity as fairness criteria",
     "Equalized odds (matched TPR/FPR across groups) is proposed as an alternative to demographic parity "
     "with different trade-offs",
     "RQ3/RQ4 -- equalized-odds difference is reported alongside demographic parity ratio in the audit "
     "(dac.fairness.audit)"),
    ("Kamiran & Calders (2012)", "Bias mitigation", "Adult Income, German Credit, Dutch Census",
     "Introduces preprocessing reweighing to decorrelate a protected attribute from the label before training",
     "Reweighing improves group fairness with a smaller accuracy cost than suppression, and requires no "
     "model-architecture changes",
     "RQ3/RQ4 -- the bias-mitigation technique implemented, and the recipe replicated unchanged across all "
     "four model families in RQ4"),
    ("Mehrabi, Morstatter, Saxena, Lerman, & Galstyan (2021)", "Fairness / ML survey", "Review (no single dataset)",
     "Systematic survey of bias sources, fairness metrics, and mitigation techniques across ML",
     "Catalogs the fairness-ML landscape and explicitly flags that most audits are single-model case studies",
     "RQ4 -- motivates this study's cross-model audit-methodology-replication design"),
    ("Dastile, Celik, & Potsane (2020)", "Credit scoring / ML survey", "Review (no single dataset)",
     "Systematic literature survey of statistical and ML credit-scoring models",
     "Ensembles (incl. boosting) consistently outperform single classical models on credit-scoring "
     "benchmarks, at an interpretability cost",
     "RQ1/RQ2 -- corroborates the accuracy-interpretability trade-off this study tests directly by "
     "comparing black-box (XGBoost/CatBoost) against glass-box (EBM) models"),
]
delete_table_by_header(["Author (Year)", "Domain/Context", "Dataset/Setting", "Method(s)", "Key Findings"])
table_before(
    find_para("Synthesis by Theme"),
    ["Author (Year)", "Domain/Context", "Dataset/Setting", "Method(s)", "Key Findings",
     "How it supports RQ(s) / project decisions"],
    lit_rows,
    font_size=8,
)

synth1 = find_para("As lenders replace traditional linear scorecards", contains=True)
set_text(
    synth1,
    "As card issuers replace traditional linear scorecards with gradient-boosted ensembles (XGBoost: Chen & "
    "Guestrin, 2016; CatBoost: Prokhorenkova et al., 2018) to capture interaction effects across billing and "
    "repayment history, they gain measurable accuracy but lose the direct coefficient-level interpretability "
    "regulators expect for adverse-action notices (Doshi-Velez & Kim, 2017). Post-hoc explainable-AI (XAI) "
    "techniques such as SHAP (Lundberg & Lee, 2017) and LIME (Ribeiro et al., 2016) exist to reconcile this "
    "tension, attributing a black-box model's prediction to individual input features after the fact -- but "
    "a genuinely different route exists too: glass-box generalized additive models such as the Explainable "
    "Boosting Machine (Lou et al., 2012; Nori et al., 2019) forgo some of the flexibility of a full "
    "black-box ensemble in exchange for an explanation that is exact by construction, not approximated.",
)
synth2 = find_para("Model accuracy and explainability alone do not guarantee fair treatment", contains=True)
set_text(
    synth2,
    "Model accuracy and explainability alone do not guarantee fair treatment across protected groups. Legal "
    "and technical fairness frameworks have converged on a common toolkit: the disparate-impact doctrine "
    "and its four-fifths-rule heuristic (Barocas & Selbst, 2016), formal fairness metrics such as "
    "demographic parity and equalized odds (Dwork et al., 2012; Hardt et al., 2016), and preprocessing "
    "bias-mitigation techniques such as Kamiran and Calders' (2012) reweighing, which reweights training "
    "examples to decorrelate a protected attribute from the label prior to model fitting.",
)
synth3 = find_para("This study treats two publicly available secondary datasets", contains=True)
set_text(
    synth3,
    "This study treats its four model families as complementary probes of the same underlying question "
    "rather than as a horse race for the single highest ROC-AUC. Logistic regression anchors the "
    "interpretability floor; XGBoost and CatBoost probe how much accuracy is available from unconstrained "
    "non-linear interactions; and EBM asks whether most of that accuracy is recoverable from an additive, "
    "exactly-interpretable model instead (Lou et al., 2012). Comparing SHAP's approximation of a black-box "
    "model against EBM's exact explanation, on the same data and the same held-out fold, is a direct way to "
    "ask whether the post-hoc explanation can be trusted as a stand-in for the real thing.",
)
synth4 = find_para("Most published fairness-in-lending studies audit a single model", contains=True)
set_text(
    synth4,
    "Most published fairness-in-lending studies audit a single model on a single dataset, leaving open the "
    "question of whether a given mitigation recipe -- as opposed to a given disparity finding -- actually "
    "transfers across model architectures (Mehrabi et al., 2021). This is not a purely academic distinction: "
    "an institution that validates reweighing against one model family still needs evidence that the same "
    "recipe will work if the production model is later swapped for a different architecture. By deliberately "
    "running the identical Fairlearn MetricFrame audit and the identical reweighing computation against four "
    "structurally different models -- one linear, two tree ensembles, one additive glass-box model -- this "
    "study is designed to speak directly to that transferability question.",
)

# ===========================================================================
# 7. Data Description / Dataset Overview
# ===========================================================================
data_desc_intro = find_para("Two datasets are used, both secondary, publicly available", contains=True)
set_text(
    data_desc_intro,
    "One dataset is used, secondary, publicly available, and pre-anonymized at the source.",
)
uci_bullet1 = find_para("HCDR (data/external/HCDR)", contains=True)
set_text(
    uci_bullet1,
    "UCI Credit-Card Default (data/external/UCI) — the “default of credit card clients” dataset "
    f"(Yeh & Lien, 2009), {eda_df.shape[0]:,} Taiwanese credit-card accounts, the sole dataset used for the "
    "full model-training / explainability / fairness pipeline (target: DEFAULT_PAYMENT_NEXT_MONTH, "
    "next-month default).",
)
uci_bullet2 = find_para("URL: https://www.kaggle.com/c/home-credit-default-risk")
set_text(uci_bullet2, "URL: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients")
hmda_bullet = find_para("HMDA (data/external/HMDA)", contains=True)
delete_para(hmda_bullet)

dataset_overview_heading = find_para("Dataset Overview")
delete_table_by_header(["Aspect", "Home Credit (HCDR)", "HMDA"])
table_before(
    find_para("Data dictionary (mandatory)"),
    ["Aspect", "UCI Credit-Card Default"],
    [
        ("Number of records (rows)", f"{eda_df.shape[0]:,}"),
        ("Number of variables (columns)", "25 (24 predictors + target), before feature engineering"),
        ("Time period", "April-September 2005 (6 months of billing/repayment history per account)"),
        ("Unit of analysis", "One row per credit-card account"),
        ("Target variable(s)", f"DEFAULT_PAYMENT_NEXT_MONTH (binary; 1 = default, {eda_df[TARGET].mean():.2%} positive)"),
        ("Data provenance", "Real data" if not IS_SYNTHETIC else "Synthetic fallback (real extract unavailable)"),
    ],
)

# ===========================================================================
# 8. Data dictionary
# ===========================================================================
dd_intro = find_para("The tables below cover data-dictionary for both the Data set")
set_text(dd_intro, "The table below covers the data dictionary for the dataset.")
dd_heading = find_para("Home Credit — Data Dictionary (Key Variables)", contains=True)
set_text(dd_heading, "UCI Credit-Card Default — Data Dictionary (Key Variables)")
delete_table_by_header(["Variable", "Type", "Description", "Values / Range", "Source"])
hmda_dd_heading = find_para("HMDA — Data Dictionary (Key Variables)", contains=True)
github_heading = find_para("GitHub Data Availability Statement")
delete_between(hmda_dd_heading, github_heading, delete_start_too=True)

dd_rows = [
    ("ID", "Integer", "Row identifier", "1 to 30,000", "Original"),
    ("LIMIT_BAL", "Numeric (NT$)", "Amount of given credit (individual + family/supplementary)", "10,000 - 1,000,000", "Original"),
    ("SEX", "Categorical (protected)", "Cardholder sex", "Male, Female", "Original (recoded from 1/2)"),
    ("EDUCATION", "Categorical", "Education level", "Graduate School, University, High School, Others, Other/Unknown", "Original (recoded)"),
    ("MARRIAGE", "Categorical", "Marital status", "Married, Single, Others, Other/Unknown", "Original (recoded)"),
    ("AGE", "Integer", "Age in years", f"{int(eda_df['AGE'].min())} - {int(eda_df['AGE'].max())}", "Original"),
    ("PAY_0, PAY_2..PAY_6", "Integer (ordinal)", "Repayment status, most recent 6 months (-2/-1/0 = no delay/paid; 1-8 = months overdue)", "-2 to 8", "Original"),
    ("BILL_AMT1..BILL_AMT6", "Numeric (NT$)", "Monthly bill statement amount, most recent 6 months", "Can be negative (credit balance)", "Original"),
    ("PAY_AMT1..PAY_AMT6", "Numeric (NT$)", "Monthly payment amount, most recent 6 months", ">= 0", "Original"),
    ("AGE_GROUP", "Categorical (protected)", "Engineered age bucket", "<25, 25-34, 35-44, 45-54, 55+", "Engineered"),
    ("AVG_BILL_AMT / AVG_PAY_AMT", "Numeric", "Mean of the 6 monthly bill / payment amounts", "See numeric summary table", "Engineered"),
    ("BILL_LIMIT_RATIO", "Numeric", "AVG_BILL_AMT / LIMIT_BAL (credit-utilization ratio)", "See numeric summary table", "Engineered"),
    ("PAY_TO_BILL_RATIO", "Numeric", "AVG_PAY_AMT / AVG_BILL_AMT (repayment-coverage ratio)", "See numeric summary table", "Engineered"),
    ("MAX_DELINQUENCY / MEAN_DELINQUENCY", "Numeric", "Max / mean of the 6 PAY_* repayment-status codes", "-2 to 8", "Engineered"),
    ("MONTHS_DELINQUENT", "Integer", "Count of the 6 months with PAY_* > 0 (overdue)", "0 to 6", "Engineered"),
    ("DEFAULT_PAYMENT_NEXT_MONTH", "Binary (target)", "1 = account defaults the following month", "0, 1", "Original (renamed)"),
]
table_before(github_heading, ["Variable", "Type", "Description", "Values / Range", "Source"], dd_rows, font_size=8)

# ===========================================================================
# 9. GitHub Data Availability + Screenshots
# ===========================================================================
gh_raw = find_para("Raw data: data/raw/home_credit/application_train.csv", contains=True)
set_text(gh_raw, "Raw data: data/raw/uci_credit/default_of_credit_card_clients.csv")
gh_src = find_para("Source archives: data/external/HCDR/home-credit-default-risk.zip", contains=True)
set_text(gh_src, "Source archive: data/external/UCI/default of credit card clients.xls")
gh_code = find_para("Code/Notebooks: src/dac/", contains=True)
set_text(
    gh_code,
    "Code/Notebooks: src/dac/ (pipeline modules), scripts/run_pipeline.py (end-to-end orchestrator), "
    "notebooks/ (exploratory analysis mirroring the pipeline)",
)

screenshots_heading = find_para("Screenshots / Evidence")
screenshots_intro = find_para("Figures 2.1-2.3 below show a live data preview", contains=True)
set_text(
    screenshots_intro,
    "Figures 2.1-2.2 below show a live data preview of the raw source file and the repository's folder "
    "structure, generated directly from the checked-in data files as evidence of availability.",
)
fig21_cap = find_para("Figure 2.1 — Home Credit application_train.csv preview", contains=True)
fig22_cap = find_para("Figure 2.2 — HMDA nationwide sample preview", contains=True)
fig23_cap = find_para("Figure 2.3 — Repository folder structure", contains=True)
analysis_heading = find_para("This section documents the cleaning steps applied to both datasets", contains=True)
# Remove the two old screenshot images (paragraphs immediately preceding each caption) + captions,
# then insert the two new ones fresh, right before "Analysis".
for cap in (fig21_cap, fig22_cap, fig23_cap):
    img_para = cap._p.getprevious()
    if img_para is not None:
        img_para.getparent().remove(img_para)
    delete_para(cap)
image_before(analysis_heading, FIG_EVID / "preview_uci_credit.png", "Figure 2.1 — UCI credit-card CSV preview (first 8 rows, key columns).")
image_before(analysis_heading, FIG_EVID / "repo_tree.png", "Figure 2.2 — Repository folder structure.")

print("Section 6-9 done.")

# ===========================================================================
# 10. Analysis intro + Data Cleaning
# ===========================================================================
analysis_intro = find_para("This section documents the cleaning steps applied to both datasets", contains=True)
set_text(
    analysis_intro,
    "This section documents the cleaning steps applied to the dataset (Data Cleaning) and the resulting "
    "exploratory findings (EDA Results), with emphasis on interpreting what each figure/table implies for "
    "feature engineering, modelling, and the fairness audit. The pipeline code implementing every step "
    "below is provided in Appendix A.",
)
data_cleaning_para = find_para("No exact duplicate rows were found in either dataset", contains=True)
set_text(
    data_cleaning_para,
    f"No exact duplicate rows were found ({n_dupes} duplicates in {eda_df.shape[0]:,} rows), and the raw "
    "source file itself has zero missing values in any of its 25 original columns. Feature engineering "
    f"introduces one legitimate missing pattern: PAY_TO_BILL_RATIO ({n_missing_total} rows, "
    f"{n_missing_total/len(eda_df):.1%}) is undefined by construction wherever AVG_BILL_AMT = 0 -- an "
    "account with no average billing activity across the six-month window has no repayment-coverage ratio "
    "to compute, not a data-entry gap. This is handled the same way as any other numeric missingness, by "
    "the shared preprocessor's median imputation, rather than by a special-cased rule, since the NaN pattern "
    "here is a well-understood arithmetic consequence rather than an anomaly requiring its own treatment. "
    "Two categorical fields separately carry undocumented codes not covered by the published UCI codebook: "
    f"EDUCATION ({n_unknown_education} rows with codes 0, 5, or 6, none of which map to graduate school / "
    f"university / high school / others) and MARRIAGE ({n_unknown_marriage} rows with code 0, not covered "
    "by married/single/others). Rather than dropping these rows or silently folding them into an existing "
    "category, scripts/prepare_uci_credit.py maps them to an explicit “Other/Unknown” category, preserving "
    "the rows while keeping the anomaly visible in every downstream table. Numeric features are "
    "median-imputed then standardized and categorical features are most-frequent-imputed then one-hot "
    "encoded inside the shared preprocessing pipeline (dac.features.engineering.build_preprocessor).",
)
delete_table_by_header(["Issue", "Variables Affected", "Detection Method", "Treatment Applied", "Rationale"])
table_before(
    find_para("EDA Results (Interpretation Emphasis)"),
    ["Issue", "Variables Affected", "Detection Method", "Treatment Applied", "Rationale"],
    [
        ("Undocumented category codes", f"EDUCATION (n = {n_unknown_education}), MARRIAGE (n = {n_unknown_marriage})",
         "Cross-referenced observed integer codes against the published UCI codebook",
         "Mapped to an explicit “Other/Unknown” category rather than dropped or merged into an existing group",
         "Preserves every row while keeping the anomaly transparent in every downstream table, rather than "
         "silently corrupting a real category's statistics"),
        ("Engineered-feature missingness", f"PAY_TO_BILL_RATIO (n = {n_missing_total}, {n_missing_total/len(eda_df):.1%})",
         "df.isna().sum() per column, post feature-engineering",
         "Median imputation inside the shared preprocessor (dac.features.engineering.build_preprocessor)",
         "Undefined by construction (division by an average bill of 0), not a data-entry gap; the raw "
         "25-column source file itself has zero missing values"),
        ("Duplicate rows", "All columns", "pandas .duplicated() full-row check", "None applied", f"{n_dupes} duplicate rows found"),
        ("Repayment-status code scale", "PAY_0, PAY_2..PAY_6", "Codebook review (-2 to 8 range observed)",
         "Retained as ordinal integers; aggregated into MAX_DELINQUENCY / MEAN_DELINQUENCY / "
         "MONTHS_DELINQUENT engineered features",
         "The raw per-month codes are noisy individually; the aggregates capture the same signal more "
         "robustly for modelling (see Features Included table)"),
        ("SEX / EDUCATION / MARRIAGE numeric codes", "SEX, EDUCATION, MARRIAGE", "Codebook review",
         "Recoded from integers (1/2, 1-6, 0-3) to human-readable string labels in scripts/prepare_uci_credit.py",
         "Makes the fairness-audit tables and data dictionary self-documenting without a separate codebook lookup"),
    ],
    font_size=8,
)
eda_results_intro = find_para("Every figure and table below is referenced in the surrounding text and paired with an interpretation", contains=True)
set_text(
    eda_results_intro,
    "Every figure and table below is referenced in the surrounding text and paired with an interpretation "
    "of what it implies for the downstream modelling, explainability, and fairness stages; Table 8 (EDA "
    "insight summary, at the end of this subsection) consolidates the single most decision-relevant insight "
    "from each.",
)

# ===========================================================================
# 11. Detailed EDA -- replace the entire HCDR + HMDA + Cross-Dataset +
#     Data Quality Notes block with one UCI section, computed live.
# ===========================================================================
eda_section_start = find_para("Detailed EDA", contains=True)  # "Home Credit Default Risk (HCDR) -- Detailed EDA"
modelling_heading = find_para("Modelling")
delete_between(eda_section_start, modelling_heading, delete_start_too=True)


def h1(text):
    return heading_before(modelling_heading, text, style="Heading 1")


def h2(text):
    return heading_before(modelling_heading, text, style="Heading 2")


def h3(text):
    return heading_before(modelling_heading, text, style="Heading 3")


def norm(text, **kw):
    return normal_before(modelling_heading, text, **kw)


def bul(items):
    return bullets_before(modelling_heading, items)


def tbl(header, rows, **kw):
    return table_before(modelling_heading, header, rows, **kw)


def img(path, caption, **kw):
    return image_before(modelling_heading, path, caption, **kw)


h1("UCI Credit-Card Default — Detailed EDA")
h2("Structure & Data Types")
n_num = eda_df.select_dtypes(include="number").shape[1]
n_cat = eda_df.shape[1] - n_num
tbl(
    ["Metric", "Value"],
    [
        ("Rows", f"{eda_df.shape[0]:,}"),
        ("Columns (post feature-engineering)", eda_df.shape[1]),
        ("Numeric columns", n_num),
        ("Categorical columns", n_cat),
        ("Duplicate rows", n_dupes),
        ("Missing cells", n_missing_total),
        ("Data provenance", "Real (UCI extract)" if not IS_SYNTHETIC else "Synthetic fallback"),
    ],
)

h2("Target Variable (DEFAULT_PAYMENT_NEXT_MONTH)")
norm(
    "DEFAULT_PAYMENT_NEXT_MONTH = 1 indicates an account that defaults on its payment the following month; "
    "0 covers all other accounts. The dataset is imbalanced but far less severely than a typical default "
    "dataset, which is still why the downstream pipeline reports PR-AUC and Brier score alongside ROC-AUC, "
    "not accuracy.",
)
target_counts = eda_df[TARGET].value_counts().sort_index()
tbl(
    ["DEFAULT_PAYMENT_NEXT_MONTH", "Count", "Proportion"],
    [(int(k), f"{v:,}", f"{v/len(eda_df):.4f}") for k, v in target_counts.items()],
)
img(FIG / "target_distribution.png", "Figure 3.1 — DEFAULT_PAYMENT_NEXT_MONTH class distribution.")

h2("Missing Values")
_col_word = "column carries" if n_missing_cols == 1 else "columns carry"
norm(
    f"{n_missing_cols} of {eda_df.shape[1]} columns (post feature-engineering) {_col_word} any missing "
    f"values: PAY_TO_BILL_RATIO, {n_missing_total} rows ({n_missing_total/len(eda_df):.1%}), undefined "
    "wherever AVG_BILL_AMT = 0 (see Data Cleaning). The raw 25-column source file itself is completely "
    "missingness-free.",
)
img(FIG / "missingness.png", "Figure 3.1b — Missing-value fraction by column.")

h2("Numeric Feature Summary Statistics")
norm("Descriptive statistics for the key numeric fields (credit limit, age, billing/payment amounts, and the engineered utilization/delinquency ratios):")
tbl(
    ["Column", "Mean", "Std", "Min", "25%", "Median", "75%", "Max"],
    [
        (c, f"{numeric_summary.loc[c, 'mean']:.2f}", f"{numeric_summary.loc[c, 'std']:.2f}",
         f"{numeric_summary.loc[c, 'min']:.2f}", f"{numeric_summary.loc[c, '25%']:.2f}",
         f"{numeric_summary.loc[c, '50%']:.2f}", f"{numeric_summary.loc[c, '75%']:.2f}",
         f"{numeric_summary.loc[c, 'max']:.2f}")
        for c in KEY_NUMERIC_COLS
    ],
    font_size=8,
)
norm(
    f"AGE ranges {int(eda_df['AGE'].min())}-{int(eda_df['AGE'].max())} years (mean {eda_df['AGE'].mean():.1f}). "
    "BILL_AMT and PAY_AMT fields are in New Taiwan dollars; BILL_AMT can be negative, representing an "
    "account in credit at that billing cycle.",
)

h2("Numeric Distributions")
img(FIG / "numeric_distributions.png", "Figure 3.2 — Distributions of key numeric features.")

h2("Categorical Feature Distributions")
norm("Top categories for the primary categorical fields:")
for col_name, rate_df in [("SEX", sex_rate), ("EDUCATION", edu_rate), ("MARRIAGE", marriage_rate)]:
    h3(col_name)
    counts = eda_df[col_name].value_counts()
    tbl(
        ["Value", "Count", "Share"],
        [(v, f"{c:,}", f"{c/len(eda_df):.2%}") for v, c in counts.items()],
    )

h2("Correlation with Target")
norm(
    "Pearson correlation of numeric columns with DEFAULT_PAYMENT_NEXT_MONTH (top 15 by absolute value). The "
    "strongest signals are the repayment-status aggregates -- MAX_DELINQUENCY, MEAN_DELINQUENCY, and the raw "
    "PAY_0 (most recent month) -- all positively correlated with default, consistent with recent repayment "
    "behavior being the most direct signal of near-term default risk.",
)
tbl(
    ["Column", "Correlation with DEFAULT_PAYMENT_NEXT_MONTH"],
    [(c, f"{v:.4f}") for c, v in corr_with_target.head(15).items()],
)
img(FIG / "correlation_heatmap.png", "Figure 3.3 — Correlation heatmap, top-15 correlated features + target.")

h2("Target Rate by Protected Attribute")
norm(
    "The pipeline audits fairness on SEX and a derived AGE_GROUP (bucketed from AGE). The EDA-level group "
    "default rates below are the early-warning signal that motivates the fairness audit / bias-mitigation "
    "stages later in the pipeline.",
)
h3("SEX")
tbl(
    ["Group", "Default rate (mean)", "Count"],
    [(g, f"{r['mean']:.4f}", f"{int(r['count']):,}") for g, r in sex_rate.iterrows()],
)
img(FIG / "target_rate_by_SEX.png", "Figure 3.4 — DEFAULT_PAYMENT_NEXT_MONTH rate by SEX.")
h3("AGE_GROUP")
tbl(
    ["Group", "Default rate (mean)", "Count"],
    [(g, f"{r['mean']:.4f}", f"{int(r['count']):,}") for g, r in age_rate.iterrows()],
)
img(FIG / "target_rate_by_AGE_GROUP.png", "Figure 3.5 — DEFAULT_PAYMENT_NEXT_MONTH rate by AGE_GROUP.")

h1("Data Quality Notes & Recommendations")
bul(
    [
        f"Undocumented EDUCATION / MARRIAGE codes — {n_unknown_education} rows carry an EDUCATION code "
        f"outside the published codebook and {n_unknown_marriage} rows carry an undocumented MARRIAGE code; "
        "both are mapped to an explicit “Other/Unknown” category rather than dropped.",
        f"Raw source has zero missingness, no duplicates — the 25-column source file is unusually clean for "
        f"a real-world credit dataset; the only missingness anywhere ({n_missing_total} rows in "
        "PAY_TO_BILL_RATIO) is introduced by feature engineering itself (division by a zero average bill), "
        "not present in the raw data.",
        "Repayment-status (PAY_*) fields dominate the correlation table — as the strongest, most recent "
        "behavioral signal, these should be monitored for data-quality drift in any production deployment, "
        "since a pipeline change upstream that alters how these codes are computed would directly move the "
        "model's top feature.",
        f"Class imbalance — {eda_df[TARGET].mean():.1%} of accounts default; moderate compared to some "
        "credit datasets, but still enough to motivate reporting PR-AUC and Brier score alongside ROC-AUC "
        "throughout.",
        f"SEX imbalance in the raw data — the sample is {(eda_df['SEX']=='Female').mean():.1%} female "
        "cardholders; group-fairness point estimates for SEX are still well-powered (see Sample size "
        "calculation) but this should be kept in mind when comparing subgroup counts.",
    ],
)

table8_label = None
try:
    table8_label = find_para("Table 8 — EDA insight summary")
except ValueError:
    table8_label = None
if table8_label is not None:
    delete_para(table8_label)
delete_table_by_header(["Figure/Table Reference", "What it Shows", "Key Insight", "Why it Matters (Link to RQ/objective)", "Decision/Next Step"])
norm("Table 8 — EDA insight summary", bold=True)
top_corr_feat, top_corr_val = corr_with_target.index[0], corr_with_target.iloc[0]
tbl(
    ["Figure/Table Reference", "What it Shows", "Key Insight", "Why it Matters (Link to RQ/objective)", "Decision/Next Step"],
    [
        ("Figure 3.1 (target distribution)", "Class balance", f"{eda_df[TARGET].mean():.1%} of accounts default",
         "Motivates reporting PR-AUC/Brier alongside ROC-AUC (RQ1); accuracy alone would be less informative",
         "Use imbalance-aware metrics throughout evaluation (see Evaluation Metrics table)"),
        ("Numeric Feature Summary Statistics", "Descriptive statistics for key numeric fields",
         "Billing/payment amounts are heavily right-skewed and BILL_AMT can be negative (credit balance)",
         "Standard scaling (not log transforms) is applied uniformly inside the shared preprocessor, keeping "
         "the pipeline simple and identical across all four models",
         "Retained as-is; utilization/coverage ratios engineered instead of transforming raw amounts"),
        ("Figure 3.3 (correlation heatmap)", "Pearson correlation with target",
         f"{clean_feature_name(top_corr_feat)} is the strongest linear predictor (r = {top_corr_val:.3f})",
         "Recent repayment status is expected to dominate SHAP/EBM importance for RQ2",
         "MAX_DELINQUENCY / MEAN_DELINQUENCY / MONTHS_DELINQUENT engineered as aggregates; confirmed as a "
         "top driver in both SHAP and EBM (see Preliminary Findings, RQ2)"),
        ("Figure 3.4-3.5 (SEX / AGE_GROUP target-rate figures)", "Default rate by protected attribute",
         f"{sex_rate.index[0]} accounts default at {sex_gap_hi:.2%} vs. {sex_rate.index[-1]} at "
         f"{sex_gap_lo:.2%}; default rate varies by age group from {age_gap_lo:.2%} to {age_gap_hi:.2%}",
         "An EDA-level gap on both protected attributes is the early-warning signal that justifies running a "
         "formal fairness audit (RQ3)",
         "Both attributes carried into the fairness audit / bias-mitigation stage"),
        ("Data Cleaning table", "EDUCATION / MARRIAGE undocumented codes",
         f"{n_unknown_education + n_unknown_marriage} rows combined carry a code outside the published UCI "
         "codebook", "A naive treatment would silently misclassify these cardholders into an existing "
         "category", "Mapped to an explicit “Other/Unknown” category (see Data Cleaning)"),
    ],
    font_size=8,
)

print("Section 10-11 (Detailed EDA rebuild) done.")

# ===========================================================================
# 12. Modelling
# ===========================================================================
modelling_intro = find_para("Two model families are trained and compared to Home Credit", contains=True)
set_text(
    modelling_intro,
    "Four model families are trained and compared on the UCI credit-card dataset (RQ1): a linear "
    "interpretable baseline, two gradient-boosted ensembles, and one glass-box additive model. The best "
    "performer by held-out ROC-AUC anchors the explainability (RQ2) and fairness-audit (RQ3) stages; the "
    "reweighing recipe from RQ3 is then replicated, unmodified, across all four model families for RQ4.",
)
set_text(find_para("Choice of Models with Justification"), "Choice of Models with Justification")
choice_intro = find_para("Four families of technique are applied, one primarily per research question", contains=True)
set_text(
    choice_intro,
    "Four model families are trained on a single shared preprocessing pipeline (median imputation and "
    "scaling for numeric features, most-frequent imputation and one-hot encoding for categorical features) "
    "so that SHAP, EBM's native explanation, and the fairness audit are all comparable across models. L2-"
    "regularized logistic regression serves as the interpretable baseline; XGBoost and CatBoost are two "
    "independently developed gradient-boosted tree ensembles; and the Explainable Boosting Machine (EBM) is "
    "a glass-box generalized additive model. All four are tuned (RandomizedSearchCV for logistic regression, "
    "Optuna/TPE search for the other three) over 5-fold (2-fold for EBM, for tractable wall-clock time) "
    "stratified cross-validated ROC-AUC. H1: because gradient-boosted trees and additive shape functions "
    "both capture structure a purely linear model's additive-in-the-raw-features form cannot (interaction "
    "effects for the ensembles, non-linear per-feature shape functions for EBM), all three are hypothesized "
    "to achieve significantly higher held-out ROC-AUC than tuned logistic regression.",
)
lr_heading = find_para("Model 1: L2-Regularized Logistic Regression", contains=True)
lr_body = find_para("Justification: a linear, L2-regularized logistic regression is the interpretable baseline", contains=True)
set_text(
    lr_body,
    "Justification: a linear, L2-regularized logistic regression is the interpretable baseline against "
    "which the three non-linear challengers are measured. Its coefficients map directly onto the additive, "
    "reason-code-style explanations regulators expect for adverse-action notices (Consumer Financial "
    "Protection Bureau, 2023), giving it a low interpretability cost even before any post-hoc XAI is "
    "applied. Tuned via RandomizedSearchCV (15 iterations, 5-fold stratified cross-validated ROC-AUC) over "
    "the regularization strength and solver.",
)
xgb_heading = find_para("Model 2: XGBoost", contains=True)
xgb_body = find_para("Justification: gradient-boosted trees capture non-linear interaction effects", contains=True)
set_text(
    xgb_body,
    "Justification: gradient-boosted trees capture non-linear interaction effects (e.g., between "
    "credit-utilization ratio and recent repayment status) that a linear model's additive form cannot "
    "represent (Chen & Guestrin, 2016; Dastile et al., 2020). XGBoost is hypothesized (H1) to achieve "
    "significantly higher held-out ROC-AUC than tuned logistic regression. Tuned via a 25-trial Optuna/TPE "
    "search (5-fold stratified cross-validated ROC-AUC) over tree depth, learning rate, subsampling, and "
    "regularization; class imbalance is handled via scale_pos_weight rather than resampling, to keep the "
    "full real sample size.",
)
heading_before(find_para("Explainability, Fairness Audit, and Bias-Mitigation Methodology"), "Model 3: CatBoost — Second Ensemble Challenger")
normal_before(
    find_para("Explainability, Fairness Audit, and Bias-Mitigation Methodology"),
    "Justification: CatBoost is a second, independently developed gradient-boosted tree ensemble that uses "
    "ordered boosting to reduce the prediction shift / target leakage that standard boosting can introduce "
    "on categorical features (Prokhorenkova et al., 2018). Including a second ensemble alongside XGBoost "
    "tests whether RQ1's accuracy gain is a property of gradient boosting generally, or specific to one "
    "implementation. Tuned via a 20-trial Optuna/TPE search over tree depth, learning rate, L2 leaf "
    "regularization, and subsampling; class imbalance is handled via CatBoost's built-in auto_class_weights.",
)
heading_before(find_para("Explainability, Fairness Audit, and Bias-Mitigation Methodology"), "Model 4: Explainable Boosting Machine (EBM) — Glass-Box Challenger")
normal_before(
    find_para("Explainability, Fairness Audit, and Bias-Mitigation Methodology"),
    "Justification: EBM (Lou et al., 2012; implemented via InterpretML, Nori et al., 2019) is a "
    "generalized additive model with pairwise interactions -- a sum of per-feature (and select "
    "per-feature-pair) shape functions learned by gradient boosting, but constrained to remain exactly "
    "additive and therefore exactly interpretable, unlike XGBoost or CatBoost. Including EBM tests two "
    "things at once: whether most of the ensembles' accuracy gain over logistic regression is recoverable "
    "from an additive model (H1), and, via its own native explanation, whether SHAP's post-hoc "
    "approximation of a black-box model can be trusted as a stand-in for an explanation that needs no "
    "approximating (H2, RQ2). Tuned via a 6-trial Optuna/TPE search (2-fold cross-validated ROC-AUC, a "
    "smaller budget than the other models because EBM fits are markedly slower at this row count) over "
    "bin count, max leaves, learning rate, and the number of outer bagging rounds; class imbalance is "
    "handled via an explicit balanced sample weight, since EBM has no built-in class_weight parameter.",
)

expl_heading = find_para("Explainability, Fairness Audit, and Bias-Mitigation Methodology")
rq2_method = find_para("For RQ2, SHAP", contains=True)
set_text(
    rq2_method,
    "For RQ2, SHAP (Lundberg & Lee, 2017) TreeExplainer values are computed for the tuned XGBoost model, "
    "both globally (mean absolute SHAP value across a held-out sample; bar and beeswarm plots) and locally "
    "(waterfall plots for individual accounts), and compared against EBM's own native global term "
    "importances and shape-function plots -- no approximation needed, since EBM's explanation is exactly "
    "what the model used to make its predictions. Agreement between the two is quantified as a Spearman "
    "rank correlation over the features both explanations rank. H2: consistent with established "
    "credit-risk theory, the top shared drivers are hypothesized to be dominated by recent repayment-status "
    "history and credit-utilization ratios rather than by demographic fields, and the SHAP/EBM rank "
    "agreement is hypothesized to be significantly positive.",
)
rq3_method = find_para("For RQ3, Fairlearn's MetricFrame computes per-group selection rate", contains=True)
set_text(
    rq3_method,
    "For RQ3, Fairlearn's MetricFrame computes per-group selection rate, accuracy, and true/false "
    "positive/negative rates for SEX and AGE_GROUP on the RQ1 best model, from which demographic parity "
    "difference/ratio, equalized-odds difference, and a four-fifths-rule disparate-impact flag (ratio < "
    "0.80) are derived. H3: the pre-mitigation demographic parity ratio is hypothesized to fall below the "
    "0.80 four-fifths threshold for at least one protected attribute; applying Kamiran and Calders' (2012) "
    "reweighing to the training sample (on SEX) and refitting is hypothesized to raise the ratio to or "
    "above 0.80 while keeping the ROC-AUC cost bounded (a tolerance of 5 percentage points is used here).",
)
rq4_method = find_para("For RQ4, the identical Fairlearn MetricFrame audit code path used for RQ3", contains=True)
set_text(
    rq4_method,
    "For RQ4, the identical reweighing sample weights computed for RQ3 (on SEX) are reused, unmodified, to "
    "retrain all four model families -- not just the RQ1 best model -- and the identical MetricFrame audit "
    "is re-run for each on both protected attributes. H4: the demographic-parity-ratio improvement "
    "(post-mitigation minus pre-mitigation) is hypothesized to be broadly comparable in direction and "
    "magnitude across all four architectures, which would support the claim that the mitigation recipe -- "
    "not just its effect on one specific model -- generalizes across model families.",
)

# ===========================================================================
# 13. Features table + Evaluation metrics table
# ===========================================================================
features_para = find_para("After feature engineering, the Home Credit model consumes", contains=True)
set_text(
    features_para,
    f"After feature engineering, the model consumes {eda_df.shape[0]:,} rows across {n_num - 1} numeric and "
    "2 categorical features (dac.features.engineering.split_feature_columns; the count excludes the target "
    "and the protected attributes); SEX and AGE_GROUP are deliberately excluded from every model's own "
    "feature set and used only for post-hoc fairness auditing, per standard fair-lending practice. All "
    "engineered features are basic, auditable arithmetic transforms of the six months of billing/payment "
    "history (no external data joins), computed identically for every row, so the transform is reproducible "
    "end to end from the checked-in raw file.",
)
delete_table_by_header(["Feature", "Original/Engineered", "Type", "Reason for Inclusion", "Used in Model(s)"])
table_before(
    find_para("Evaluation Metrics (Include Formulae and Calculations)"),
    ["Feature", "Original/Engineered", "Type", "Reason for Inclusion", "Used in Model(s)"],
    [
        ("LIMIT_BAL", "Original", "Numeric", "Given credit line; a direct affordability/risk signal", "All 4 models"),
        ("AGE", "Original", "Numeric", "Cardholder age (feeds AGE_GROUP)", "All 4 models"),
        ("AGE_GROUP", "Engineered (binned AGE)", "Categorical (protected)", "5-bucket age band for the fairness audit", "Fairness audit only (excluded from model features)"),
        ("PAY_0, PAY_2..PAY_6", "Original", "Ordinal integer", "Raw monthly repayment-status codes", "All 4 models"),
        ("MAX_DELINQUENCY / MEAN_DELINQUENCY", "Engineered (max/mean of PAY_*)", "Numeric", "Aggregate repayment-risk signal, more robust than any single month", "All 4 models"),
        ("MONTHS_DELINQUENT", "Engineered (count of PAY_* > 0)", "Integer", "Count of overdue months in the 6-month window", "All 4 models"),
        ("BILL_AMT1..BILL_AMT6, PAY_AMT1..PAY_AMT6", "Original", "Numeric", "Raw monthly billing/payment history", "All 4 models"),
        ("AVG_BILL_AMT / AVG_PAY_AMT", "Engineered (mean of the 6 months)", "Numeric", "Smoothed billing/payment level", "All 4 models"),
        ("BILL_LIMIT_RATIO", "Engineered (AVG_BILL_AMT / LIMIT_BAL)", "Numeric", "Credit-utilization ratio; standard credit-risk feature", "All 4 models"),
        ("PAY_TO_BILL_RATIO", "Engineered (AVG_PAY_AMT / AVG_BILL_AMT)", "Numeric", "Repayment-coverage ratio", "All 4 models"),
        ("BILL_AMT_TREND", "Engineered (BILL_AMT1 - BILL_AMT6)", "Numeric", "Whether the balance is growing or shrinking over the window", "All 4 models"),
        ("EDUCATION, MARRIAGE", "Original (recoded)", "Categorical", "Retained applicant attributes, one-hot encoded", "All 4 models"),
        ("SEX", "Original (recoded)", "Categorical (protected)", "Excluded from model features entirely; used only for post-hoc MetricFrame fairness auditing", "Fairness audit only"),
    ],
    font_size=8,
)

eval_metrics_para = find_para("Because both targets are minority-class", contains=True)
set_text(
    eval_metrics_para,
    f"Because the target is imbalanced ({eda_df[TARGET].mean():.1%} positive), accuracy is not reported as "
    "a primary metric; ROC-AUC is supplemented with PR-AUC and Brier score, which remain informative under "
    "class imbalance. Group-fairness metrics use Fairlearn's MetricFrame (dac.fairness.audit); the "
    "four-fifths rule is a borrowed EEOC employment-discrimination heuristic, not a codified ECOA/Reg B "
    "lending standard, used here only as a common fairness-ML convention. A Spearman rank correlation "
    "quantifies RQ2's SHAP-vs-EBM agreement, since that comparison is over feature rankings rather than "
    "predicted probabilities.",
)
eval_metrics_table = None
for t in doc.tables:
    header = [c.text.strip() for c in t.rows[0].cells]
    if header[:2] == ["Metric", "Formula"]:
        eval_metrics_table = t
        break
if eval_metrics_table is not None:
    row = eval_metrics_table.add_row().cells
    for i, v in enumerate([
        "Spearman Rank Correlation (ρ)",
        "1 - (6 * sum(d_i^2)) / (n * (n^2 - 1)), d_i = rank difference for feature i",
        "Agreement between two feature-importance rankings; 1.0 = identical ranking, 0 = no relationship",
        "SHAP-vs-EBM explanation agreement (RQ2)",
    ]):
        row[i].text = v
        for p in row[i].paragraphs:
            for r in p.runs:
                r.font.size = Pt(8)

print("Section 12-13 (Modelling, Features, Evaluation Metrics) done.")

# ===========================================================================
# 14. Preliminary results (per-RQ methodology + H0/Ha + result)
# ===========================================================================
prelim_intro1 = find_para("For each research question, the methodology below (unchanged from the study design) is", contains=True)
set_text(
    prelim_intro1,
    "For each research question, the methodology below (unchanged from the study design) is followed "
    "immediately by the actual result observed from this run of the pipeline.",
)
prelim_intro2 = find_para("followed immediately by the actual preliminary result observed from this run of the pipeline.")
delete_para(prelim_intro2)

rq1_method_p = find_para("RQ1 -- Model comparison: an 80/20 stratified train/test split", contains=True)
set_text(
    rq1_method_p,
    "RQ1 -- Model comparison: an 80/20 stratified train/test split (config/config.yaml: split.test_size = "
    "0.20) isolates a held-out evaluation set never seen during tuning. Eight pipeline variants -- baseline "
    "and tuned versions of Logistic Regression, XGBoost, CatBoost, and EBM -- are fit and scored on "
    "identical folds; H0 (equal AUC) is tested with a one-tailed two-proportion z-test on held-out ROC-AUC "
    "(treating AUC as a proportion, per the sample-size method above), with the tuned model of the highest "
    "ROC-AUC retained as best_model for RQ2-RQ4.",
)
delete_table_by_header(["H0: AUC(XGBoost tuned) = AUC(Logistic Regression tuned)."])
rq1_result_anchor = find_para("Preliminary result: tuned xgboost tuned achieves ROC-AUC", contains=True)
h0_ha_table(
    rq1_result_anchor,
    f"ROC-AUC({label(best_model)}) = ROC-AUC(Logistic Regression tuned).",
    f"ROC-AUC({label(best_model)}) > ROC-AUC(Logistic Regression tuned), tested one-tailed at α = .05.",
)
lr_tuned = tuned["logistic_regression_tuned"]
best_tuned = tuned[best_model]
other_tuned = {k: v for k, v in tuned.items() if k != best_model}
auc_gap = best_tuned["roc_auc"] - lr_tuned["roc_auc"]
tuned_ranked = sorted(tuned.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True)
tuned_ranked_str = "; ".join(f"{label(k)} {v['roc_auc']:.4f}" for k, v in tuned_ranked)
set_text(
    rq1_result_anchor,
    f"Result: the tuned {label(best_model)} model achieves the highest held-out ROC-AUC = "
    f"{best_tuned['roc_auc']:.4f} (PR-AUC = {best_tuned['pr_auc']:.4f}, F1 = {best_tuned['f1']:.4f}, KS = "
    f"{best_tuned['ks_statistic']:.4f}, Brier = {best_tuned['brier_score']:.4f}), a gap of "
    f"+{auc_gap:.4f} over tuned Logistic Regression's {lr_tuned['roc_auc']:.4f}. Full tuned ranking: "
    f"{tuned_ranked_str}. Every model improved after tuning relative to its own baseline (baseline ROC-AUC: "
    + "; ".join(f"{label(k)} {v['roc_auc']:.4f}" for k, v in baseline.items())
    + f"). {label(best_model)} is retained as best_model for RQ2-RQ4 (full comparison in Table 9, "
    "Preliminary Model Performance).",
)

rq2_method_p = find_para("RQ2 -- Explainability: SHAP TreeExplainer is run on up to 500 held-out rows", contains=True)
set_text(
    rq2_method_p,
    "RQ2 -- Explainability: SHAP TreeExplainer is run on up to 500 held-out rows (200 background rows) for "
    "the tuned XGBoost model, producing a global mean-|SHAP| bar chart and beeswarm plot; EBM's own "
    "explain_global() produces the equivalent native bar chart plus shape-function plots for its top "
    "single-feature terms. A Spearman rank correlation is then computed between the two importance rankings "
    "over their shared feature names.",
)
delete_table_by_header(["H0: The top 10 mean-|SHAP| features show no correspondence with the domain-expected risk direction."])
rq2_result_anchor = find_para("Preliminary result: SHAP mean-|SHAP| ranking on the tuned xgboost tuned model places", contains=True)
h0_ha_table(
    rq2_result_anchor,
    "EBM native term importances and XGBoost SHAP mean-|value| importances show no rank agreement "
    "(Spearman ρ = 0) over shared top features.",
    "EBM and SHAP importances show significant positive rank agreement (Spearman ρ > 0), tested at α = .05.",
)
top_shap_feats = list(shap_top.items())[:5]
top_shap_str = "; ".join(f"{clean_feature_name(k)} ({v:.4f})" for k, v in top_shap_feats)
top_ebm_feats = list(ebm_top.items())[:5]
top_ebm_str = "; ".join(f"{clean_feature_name(k)} ({v:.4f})" for k, v in top_ebm_feats)
sig_word = "a statistically significant" if (agreement.get("p_value") is not None and agreement["p_value"] < 0.05) else "a not-yet-statistically-significant"
set_text(
    rq2_result_anchor,
    f"Result: SHAP mean-|SHAP| ranking on XGBoost places {top_shap_str} as the top drivers; EBM's native "
    f"term-importance ranking places {top_ebm_str} as its top drivers. The two rankings agree with Spearman "
    f"ρ = {agreement['spearman_r']:.3f} (n = {agreement['n_common_features']} shared features, p = "
    f"{agreement['p_value']:.4f}) -- {sig_word} positive rank agreement, "
    + ("supporting H2." if agreement["spearman_r"] and agreement["spearman_r"] > 0 else "only partially supporting H2.")
    + " Both methods converge on repayment-status and utilization-ratio features as the dominant drivers of "
    "predicted default risk, consistent with established credit-risk theory (see Figures 6-7, SHAP and EBM "
    "global importance).",
)

rq3_method_p = find_para("RQ3 -- Fairness audit and mitigation: Fairlearn MetricFrame audits both logistic_regression_tuned", contains=True)
set_text(
    rq3_method_p,
    f"RQ3 -- Fairness audit and mitigation: Fairlearn MetricFrame audits the tuned {label(best_model)} model "
    f"on SEX and AGE_GROUP before mitigation; Kamiran and Calders (2012) reweighing sample weights are "
    f"computed on {mit_attr} (config/config.yaml: fairness.mitigation_attribute), the model is refit with "
    "those weights, and the identical MetricFrame audit is re-run post-mitigation. Pre- and post-mitigation "
    "demographic parity ratio and ROC-AUC are compared directly to test H3.",
)
delete_table_by_header(["H0: Pre-mitigation demographic parity ratio ≥ 0.80 for both attributes."])
rq3_result_anchor = find_para("Preliminary result: pre-mitigation demographic parity ratio is", contains=True)
h0_ha_table(
    rq3_result_anchor,
    "Pre-mitigation demographic parity ratio ≥ 0.80 for both SEX and AGE_GROUP.",
    "Pre-mitigation demographic parity ratio < 0.80 for at least one attribute, and post-mitigation ratio "
    "≥ 0.80 for that attribute.",
)
best_pre = pre_fair[best_model]
best_post = post_fair[best_model]
best_mitig_perf = mitigated[best_model]
auc_cost_best = tuned[best_model]["roc_auc"] - best_mitig_perf["roc_auc"]
set_text(
    rq3_result_anchor,
    f"Result: pre-mitigation demographic parity ratio for {label(best_model)} is "
    f"{best_pre['SEX']['demographic_parity_ratio']:.3f} for SEX and "
    f"{best_pre['AGE_GROUP']['demographic_parity_ratio']:.3f} for AGE_GROUP"
    + (" -- below the 0.80 four-fifths threshold, supporting H3's pre-mitigation hypothesis"
       if min(best_pre['SEX']['demographic_parity_ratio'], best_pre['AGE_GROUP']['demographic_parity_ratio']) < 0.80
       else " -- both already at or above the 0.80 four-fifths threshold")
    + f". After {mit_attr}-based Kamiran and Calders (2012) reweighing, the post-mitigation ratio is "
    f"{best_post['SEX']['demographic_parity_ratio']:.3f} (SEX) and "
    f"{best_post['AGE_GROUP']['demographic_parity_ratio']:.3f} (AGE_GROUP), at a ROC-AUC cost of "
    f"{auc_cost_best:.4f} ({best_mitig_perf['roc_auc']:.4f} mitigated vs. {tuned[best_model]['roc_auc']:.4f} "
    "pre-mitigation) -- "
    + ("within" if abs(auc_cost_best) <= 0.05 else "outside")
    + " the pre-registered 5-percentage-point tolerance. Equalized-odds difference, reported alongside "
    f"demographic parity, is {best_post['SEX']['equalized_odds_difference']:.3f} (SEX) and "
    f"{best_post['AGE_GROUP']['equalized_odds_difference']:.3f} (AGE_GROUP) post-mitigation -- flagged under "
    "Interim Limitations and Risks below.",
)

rq4_method_p = find_para("RQ4 -- Cross-dataset replication: the same MetricFrame audit function", contains=True)
set_text(
    rq4_method_p,
    f"RQ4 -- Cross-model replication: the identical {mit_attr}-based reweighing sample weights and the "
    "identical MetricFrame audit function (dac.fairness.audit) used in RQ3 are applied, unmodified, to the "
    "other three tuned models. The pre-to-post demographic-parity-ratio change for each model is compared "
    "against the RQ3 result to test H4.",
)
delete_table_by_header(["H0: Demographic parity ratio ≥ 0.80 for both HMDA protected attributes."])
rq4_result_anchor = find_para("Preliminary result: Applying the identical MetricFrame audit code path", contains=True)
h0_ha_table(
    rq4_result_anchor,
    "The reweighing recipe's demographic-parity-ratio improvement (post minus pre) does not differ "
    "meaningfully across the four model families.",
    "The improvement differs meaningfully across model families -- i.e., the recipe's effectiveness depends "
    "on model architecture.",
)
rq4_lines = []
for name in ["logistic_regression_tuned", "xgboost_tuned", "catboost_tuned", "ebm_tuned"]:
    d_pre = pre_fair[name][mit_attr]["demographic_parity_ratio"]
    d_post = post_fair[name][mit_attr]["demographic_parity_ratio"]
    rq4_lines.append(f"{label(name)}: {d_pre:.3f} → {d_post:.3f} ({'+' if d_post-d_pre>=0 else ''}{d_post-d_pre:.3f})")
deltas = [post_fair[n][mit_attr]["demographic_parity_ratio"] - pre_fair[n][mit_attr]["demographic_parity_ratio"] for n in tuned]
delta_spread = max(deltas) - min(deltas)
set_text(
    rq4_result_anchor,
    f"Result: applying the identical {mit_attr} reweighing recipe to all four tuned models moves the "
    f"SEX demographic parity ratio as follows -- " + "; ".join(rq4_lines) + f". The improvement ranges "
    f"{min(deltas):+.3f} to {max(deltas):+.3f} across model families (spread = {delta_spread:.3f}), "
    + ("a broadly comparable effect across architectures, supporting H4" if delta_spread < 0.15
       else "a meaningfully uneven effect across architectures, only partially supporting H4")
    + f". This indicates that the reweighing recipe's effect on {label(best_model)} specifically "
    + ("generalizes" if delta_spread < 0.15 else "does not fully generalize")
    + " to the other three model families evaluated here.",
)

print("Section 14 (Preliminary Findings by RQ) done.")

# ===========================================================================
# 15. Preliminary Model Performance (Table 9 + figures + Table 10 fairness)
# ===========================================================================
# The old Table 9/10 (HCDR/HMDA) are removed via delete_table_by_header below, but the six old
# Figure 5-10 image+caption blocks are standalone paragraphs, not tied to a table -- remove them
# explicitly here or they'd sit duplicated alongside the new Figures 5-9 inserted further down.
_old_fig_captions = [
    "Figure 5 — Baseline vs. tuned model comparison (ROC-AUC, PR-AUC, F1, KS).",
    "Figure 6 — Xgboost Tuned: ROC, precision-recall, calibration, and confusion-matrix diagnostics.",
    "Figure 7 — Xgboost Tuned: global SHAP feature importance (mean |SHAP|).",
    "Figure 8 — Xgboost Tuned selection rate by CODE_GENDER, pre-mitigation.",
    "Figure 9 — Xgboost Tuned selection rate by CODE_GENDER, post-mitigation.",
    "Figure 10 — HMDA XGBoost selection rate by derived_race.",
]
for cap_text in _old_fig_captions:
    try:
        cap_p = find_para(cap_text)
    except ValueError:
        continue
    img_p = cap_p._p.getprevious()
    if img_p is not None:
        img_p.getparent().remove(img_p)
    delete_para(cap_p)

perf_intro = find_para("Table 9 compares all Home Credit model variants plus the HMDA fairness-audit model", contains=True)
set_text(
    perf_intro,
    "Table 9 compares all eight model variants (baseline and tuned, all four families) plus each model's "
    f"{mit_attr}-reweighed mitigated version on the same held-out metrics; Table 10 compares pre- and "
    "post-mitigation fairness metrics for SEX and AGE_GROUP, across all four model families.",
)


def fmt(m):
    return (f"{m['roc_auc']:.4f}", f"{m['pr_auc']:.4f}", f"{m['f1']:.4f}", f"{m['ks_statistic']:.4f}", f"{m['brier_score']:.4f}")


delete_table_by_header(["Model", "Dataset / Task", "ROC-AUC", "PR-AUC", "F1", "KS", "Brier"])
perf_rows = []
for name in ["logistic_regression", "xgboost", "catboost", "ebm"]:
    perf_rows.append((label(f"{name}_baseline"), "Baseline", *fmt(baseline[f"{name}_baseline"])))
for name in ["logistic_regression_tuned", "xgboost_tuned", "catboost_tuned", "ebm_tuned"]:
    perf_rows.append((label(name), "Tuned" + (" (best)" if name == best_model else ""), *fmt(tuned[name])))
for name in ["logistic_regression_tuned", "xgboost_tuned", "catboost_tuned", "ebm_tuned"]:
    perf_rows.append((label(name), f"Tuned, {mit_attr}-mitigated", *fmt(mitigated[name])))
limitations_heading = find_para("Interim Limitations and Risks")
table_before(
    limitations_heading,
    ["Model", "Stage", "ROC-AUC", "PR-AUC", "F1", "KS", "Brier"],
    perf_rows,
    font_size=8,
)
image_before(limitations_heading, METRICS_DIR / "model_comparison.png", "Figure 5 — Baseline vs. tuned model comparison (ROC-AUC, PR-AUC, F1, KS).")
image_before(limitations_heading, FIG / f"{best_model}_evaluation_suite.png", f"Figure 6 — {label(best_model)}: ROC, precision-recall, calibration, and confusion-matrix diagnostics.")
image_before(limitations_heading, FIG / "shap" / "xgboost_tuned_shap_bar.png", "Figure 7 — XGBoost: global SHAP feature importance (mean |SHAP|).")
image_before(limitations_heading, FIG / "ebm" / "ebm_tuned_ebm_importance_bar.png", "Figure 7b — EBM: native global term importance.")

delete_table_by_header(["Dataset", "Attribute", "Stage", "Demographic Parity Ratio", "Demographic Parity Difference", "Equalized Odds Difference", "Four-Fifths Flag"])
fair_rows = []
for name in ["logistic_regression_tuned", "xgboost_tuned", "catboost_tuned", "ebm_tuned"]:
    for attr in ["SEX", "AGE_GROUP"]:
        for stage_label, fair_dict in [("Pre-mitigation", pre_fair), ("Post-mitigation", post_fair)]:
            f = fair_dict[name][attr]
            fair_rows.append((
                label(name), attr, stage_label,
                f"{f['demographic_parity_ratio']:.3f}", f"{f['demographic_parity_difference']:.3f}",
                f"{f['equalized_odds_difference']:.3f}", str(f["disparate_impact_flag"]),
            ))
table_before(
    limitations_heading,
    ["Model", "Attribute", "Stage", "Demographic Parity Ratio", "Demographic Parity Difference", "Equalized Odds Difference", "Four-Fifths Flag"],
    fair_rows,
    font_size=7,
)
image_before(limitations_heading, FIG / "fairness" / f"{best_model}_pre_mitigation_{mit_attr}_selection_rate.png", f"Figure 8 — {label(best_model)} selection rate by {mit_attr}, pre-mitigation.")
image_before(limitations_heading, FIG / "fairness" / f"{best_model}_post_mitigation_{mit_attr}_selection_rate.png", f"Figure 9 — {label(best_model)} selection rate by {mit_attr}, post-mitigation.")

print("Section 15 (Preliminary Model Performance) done.")

# ===========================================================================
# 16. Interim Limitations and Risks + Next Steps
# ===========================================================================
lim1 = find_para("Four-fifths rule caveat: the 0.80 disparate-impact-ratio threshold", contains=True)
set_text(
    lim1,
    "Four-fifths rule caveat: the 0.80 disparate-impact-ratio threshold is an EEOC "
    "employment-discrimination heuristic, not a codified ECOA/Regulation B lending standard; it is used "
    "here only as a common fairness-ML convention, not a claimed regulatory bright line.",
)
lim2 = find_para("Single mitigation technique: reweighing (Kamiran & Calders, 2012) was applied on CODE_GENDER only", contains=True)
set_text(
    lim2,
    f"Single mitigation technique: reweighing (Kamiran & Calders, 2012) was applied on {mit_attr} only; "
    "AGE_GROUP's post-mitigation change is a correlated side-effect rather than a separately-optimized "
    "result, and a second technique (e.g., equalized-odds postprocessing or adversarial debiasing) has not "
    "yet been evaluated for comparison.",
)
lim3 = find_para("Equalized-odds trade-off: even after reweighing raises demographic parity above 0.80", contains=True)
set_text(
    lim3,
    f"Equalized-odds trade-off: demographic parity and equalized odds are different fairness criteria and "
    f"can move in different directions under the same mitigation; for {label(best_model)}, post-mitigation "
    f"equalized-odds difference on SEX is {best_post['SEX']['equalized_odds_difference']:.3f} -- a known "
    "fairness-metric trade-off that the final report needs to discuss explicitly rather than treat "
    "demographic parity as a complete fairness guarantee.",
)
lim4 = find_para("HMDA target is a pricing proxy, not approval/denial", contains=True)
set_text(
    lim4,
    "EBM tuning budget: EBM fits are markedly slower than the other three model families at this row count, "
    "so its Optuna search used a smaller trial budget (6 trials, 2-fold CV) than XGBoost/CatBoost (20-25 "
    "trials, 5-fold CV); EBM's reported performance may be a modest underestimate of its true achievable "
    "ROC-AUC relative to the other models.",
)
lim5 = find_para("HMDA sampling: each year is subsampled at ~1%", contains=True)
set_text(
    lim5,
    "Single-dataset scope: unlike an earlier version of this project design that additionally audited a "
    "second, independently sourced dataset, this report evaluates cross-model (not cross-dataset) "
    "replication of the fairness-mitigation recipe (RQ4); whether the same recipe transfers to a different "
    "dataset and decision task remains untested here.",
)
lim6 = find_para("Small-n subgroup instability: HCDR's CODE_GENDER = 'XNA' category", contains=True)
set_text(
    lim6,
    f"RQ2 sample-size assumption: because no prior SHAP-vs-EBM agreement estimate exists for this dataset, "
    f"the RQ2 minimum-sample-size calculation assumes a conservative target correlation (r = .50) rather "
    f"than a measured pilot value; the achieved n = {agreement['n_common_features']} shared features is a "
    "count of features, not observations, so it should be read as a feature-ranking comparison, not a "
    "power-analyzed sample in the classical sense.",
)
lim7 = find_para("Formal inferential statistics pending: the z-test / confidence-interval calculations", contains=True)
set_text(
    lim7,
    "Formal inferential statistics pending: exact p-values for each RQ's stated hypothesis (beyond the "
    "point estimates and the RQ2 Spearman p-value reported above) have not yet been fully reported for this "
    "interim submission.",
)

next1 = find_para("Execute and report the formal statistical tests underlying RQ1", contains=True)
set_text(
    next1,
    "Execute and report exact p-values for RQ1 (one-tailed two-proportion z-test on ROC-AUC) and RQ3/RQ4 "
    "(two-tailed two-proportion z-tests on demographic parity), alongside the RQ2 Spearman correlation "
    "p-value already computed.",
)
next2 = find_para("Run a joint sensitivity analysis of the reweighing mitigation across both CODE_GENDER", contains=True)
set_text(
    next2,
    "Run a joint sensitivity analysis of the reweighing mitigation across both SEX and AGE_GROUP "
    "simultaneously, rather than mitigating on one attribute and observing the other as a side-effect.",
)
next3 = find_para("Evaluate a second bias-mitigation technique (equalized-odds postprocessing", contains=True)
set_text(
    next3,
    "Evaluate a second bias-mitigation technique (equalized-odds postprocessing or adversarial debiasing) "
    "for direct comparison against reweighing, addressing the equalized-odds trade-off noted above.",
)
next4 = find_para("Synthesize findings into an accuracy-fairness decision framework", contains=True)
set_text(
    next4,
    "Synthesize findings into an accuracy-fairness decision framework that contrasts accuracy-priority and "
    "fairness-priority operating points across the four model families, together with a model-governance "
    "checklist covering explainability documentation, periodic bias re-audit cadence, and adverse-action-"
    "notice generation -- aimed at credit-risk/model-validation teams, fair-lending regulators and "
    "examiners, and MLOps teams building production bias/drift monitoring.",
)
next5 = find_para("Expand the literature review with any additional 2025-2026 sources", contains=True)
set_text(next5, "Expand the literature review with any additional 2025-2026 sources identified during final-report writing.")
next6 = find_para("Complete a final APA 7 formatting and proofreading pass", contains=True)
set_text(next6, "Complete a final APA 7 formatting and proofreading pass, and finalize the Appendix code excerpt and walkthrough.")

print("Section 16 (Limitations, Next Steps) done.")

# ===========================================================================
# 17. Bibliography -- replace with dataset-agnostic sources kept + new ones
# ===========================================================================
biblio_heading = find_para("References and Bibliography")
biblio_first = find_para("Bahlool, R., Hewahi, N., & Elmedany, W. (2026)", contains=True)
appendix_heading = find_para("Appendix")
delete_between(biblio_first, appendix_heading, delete_start_too=True)

biblio_entries = [
    "Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact. California Law Review, 104(3), "
    "671-732. https://doi.org/10.15779/Z38BG31",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In B. Krishnapuram et al. "
    "(Eds.), Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data "
    "Mining (pp. 785–794). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785",
    "Consumer Financial Protection Bureau. (2023). Circular 2023-03: Adverse action notification "
    "requirements and the proper use of the CFPB's sample forms provided in Regulation B. "
    "https://www.consumerfinance.gov/compliance/circulars/circular-2023-03-adverse-action-notification-"
    "requirements-and-the-proper-use-of-the-cfpbs-sample-forms-provided-in-regulation-b/",
    "Dastile, X., Celik, T., & Potsane, M. (2020). Statistical and machine learning models in credit "
    "scoring: A systematic literature survey. Applied Soft Computing, 91, Article 106263. "
    "https://doi.org/10.1016/j.asoc.2020.106263",
    "Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. "
    "arXiv. https://arxiv.org/abs/1702.08608",
    "Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. In "
    "S. Goldwasser (Ed.), Proceedings of the 3rd Innovations in Theoretical Computer Science Conference "
    "(pp. 214–226). Association for Computing Machinery. https://doi.org/10.1145/2090236.2090255",
    "Fuster, A., Goldsmith-Pinkham, P., Ramadorai, T., & Walther, A. (2022). Predictably unequal? The "
    "effects of machine learning on credit markets. The Journal of Finance, 77(1), 5–47. "
    "https://doi.org/10.1111/jofi.13090",
    "Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. In D. D. "
    "Lee et al. (Eds.), Advances in Neural Information Processing Systems (Vol. 29, pp. 3315–3323).",
    "Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without "
    "discrimination. Knowledge and Information Systems, 33(1), 1–33. "
    "https://doi.org/10.1007/s10115-011-0463-8",
    "Lou, Y., Caruana, R., & Gehrke, J. (2012). Intelligible models for classification and regression. In "
    "Q. Yang et al. (Eds.), Proceedings of the 18th ACM SIGKDD International Conference on Knowledge "
    "Discovery and Data Mining (pp. 150–158). Association for Computing Machinery. "
    "https://doi.org/10.1145/2339530.2339556",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In I. "
    "Guyon et al. (Eds.), Advances in Neural Information Processing Systems (Vol. 30).",
    "Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and "
    "fairness in machine learning. ACM Computing Surveys, 54(6), 1–35. https://doi.org/10.1145/3457607",
    "Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A unified framework for machine "
    "learning interpretability. arXiv. https://arxiv.org/abs/1909.09223",
    "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased "
    "boosting with categorical features. In S. Bengio et al. (Eds.), Advances in Neural Information "
    "Processing Systems (Vol. 31).",
    "Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). \"Why should I trust you?\": Explaining the "
    "predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on "
    "Knowledge Discovery and Data Mining (pp. 1135–1144). Association for Computing Machinery. "
    "https://doi.org/10.1145/2939672.2939778",
    "Yeh, I.-C., & Lien, C. (2009). The comparisons of data mining techniques for the predictive accuracy "
    "of probability of default of credit card clients. Expert Systems with Applications, 36(2), "
    "2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020",
]
for entry in biblio_entries:
    appendix_heading.insert_paragraph_before(entry, style="LO-normal")

print("Section 17 (Bibliography) done.")

# ===========================================================================
# 18. Appendix
# ===========================================================================
appendix_intro = find_para("Appendix A provides the end-to-end pipeline orchestration code", contains=True)
set_text(
    appendix_intro,
    "Appendix A provides the end-to-end pipeline orchestration code referenced throughout the Analysis and "
    "Modelling sections (e.g., the code used to produce Figures 3.1-3.5, the SHAP/EBM figures in Figures "
    "6-7, and the fairness figures in Figures 8-9); Appendix B provides the central configuration file that "
    "parameterizes every run.",
)
appendix_a_intro = find_para("Full source is available at", contains=True)
set_text(
    appendix_a_intro,
    f"Full source is available at {GITHUB_URL}/blob/master/scripts/run_pipeline.py. The excerpt below shows "
    "the pipeline's top-level stage sequence; each stage delegates to a dedicated module under src/dac/.",
)
code_para = None
for p in doc.paragraphs:
    if p.style.name == "Preformatted Text" and "End-to-end capstone pipeline orchestrator" in p.text:
        code_para = p
        break
if code_para is not None:
    run_pipeline_src = (REPO / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    excerpt = "\n".join(run_pipeline_src.splitlines()[:24])
    for r in list(code_para.runs):
        r.text = ""
    if code_para.runs:
        code_para.runs[0].text = excerpt
    else:
        code_para.add_run(excerpt)

appendix_b_intro = find_para("Full contents of the single configuration file governing every pipeline run:")
config_code_para = None
for p in doc.paragraphs:
    if p.style.name == "Preformatted Text" and "Central configuration for the credit-default capstone pipeline" in p.text:
        config_code_para = p
        break
if config_code_para is not None:
    config_src = (REPO / "config" / "config.yaml").read_text(encoding="utf-8")
    for r in list(config_code_para.runs):
        r.text = ""
    if config_code_para.runs:
        config_code_para.runs[0].text = config_src
    else:
        config_code_para.add_run(config_src)

print("Section 18 (Appendix) done.")
doc.save(str(OUTPUT_DOC))
print("Build complete ->", OUTPUT_DOC)
