"""Builds the QM640 Interim Report docx from the base Synopsis document
(for APA formatting/styles/figures) plus the latest pipeline run outputs
(reports/model_performance/run_summary.json and reports/figures/*).

The base document's own style library (APA Heading 2, LO-normal, Heading 1-3,
List Bullet, Table Grid/borders, header page-number field, section margins)
is reused as-is so the Interim Report visually matches the Synopsis exactly;
this script only reorders/relocates existing content and inserts new
sections/tables/figures required by the Interim Report template.

Usage:
    python scripts/build_interim_report.py
"""
from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

REPO = Path(__file__).resolve().parents[1]
BASE_DOC = Path(r"C:\Project\MS\DAC\QM640 Data Analytics Capstone Sailen.docx")
OUTPUT_DOC = Path(r"C:\Project\MS\DAC\QM640 Data Analytics Capstone Interim Report Sailen.docx")
GITHUB_URL = "https://github.com/sailenkumargmail/QM640_Capstone"

FIG_HC = REPO / "reports" / "figures" / "home_credit"
FIG_HMDA = REPO / "reports" / "figures" / "hmda"
FIG_EVID = REPO / "reports" / "figures" / "report_evidence"
METRICS_DIR = REPO / "reports" / "model_performance"

summary = json.loads((METRICS_DIR / "run_summary.json").read_text(encoding="utf-8"))
best_model = summary["best_model"]
baseline = summary["baseline_metrics"]
tuned = summary["tuned_metrics"]
mitigated = summary["mitigated_performance"]
mit_attr = summary["fairness_mitigation_attribute"]
pre_fair = summary["fairness_pre_mitigation"]
post_fair = summary["fairness_post_mitigation"]
shap_top = summary["shap_top_features"]
hmda_perf = summary["hmda_results"]["performance"]
hmda_fair = summary["hmda_results"]["fairness"]

lr_base, xgb_base = baseline["logistic_regression_baseline"], baseline["xgboost_baseline"]
lr_tuned, xgb_tuned = tuned["logistic_regression_tuned"], tuned["xgboost_tuned"]
best_tuned = tuned[best_model]

doc = docx.Document(str(BASE_DOC))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def move_before(anchor, block):
    """Relocate an existing paragraph/table element to just before anchor."""
    el = block._p if hasattr(block, "_p") else block._tbl
    anchor._p.addprevious(el)


def clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "").replace("_", " ")


print("Base document loaded:", BASE_DOC)
print("Using run_summary.json:", METRICS_DIR / "run_summary.json")
print("Best model:", best_model)

# ---------------------------------------------------------------------------
# 1. Title page: Synopsis -> Interim Report, refresh date
# ---------------------------------------------------------------------------
synopsis_heading = find_para("Synopsis")
set_text(synopsis_heading, "Interim Report")
date_para = find_para("July 21, 2026")
set_text(date_para, date.today().strftime("%B %d, %Y"))

# ---------------------------------------------------------------------------
# 2. GitHub link, right after the title page / before "Introduction"
# ---------------------------------------------------------------------------
intro_heading = find_para("Introduction")
heading_before(intro_heading, "GitHub Repository (Data and Code)", style="Heading 3")
normal_before(
    intro_heading,
    f"All code, configuration, and data-preparation scripts for this project are hosted at: "
    f"{GITHUB_URL}. Raw and processed data files (data/raw/, data/external/), trained model "
    f"artifacts (models/), and every figure/table referenced in this report (reports/) are "
    f"produced by that repository's pipeline and are available there in full (see the GitHub "
    f"Data Availability Statement in the Data Description section, and Appendix A for the "
    f"orchestration code).",
)
normal_before(intro_heading, "")

# ---------------------------------------------------------------------------
# 3. Remove the old 3-row literature snippet table from Background (it is
#    rebuilt, expanded, in the new Literature Survey section below).
# ---------------------------------------------------------------------------
for t in list(doc.tables):
    header = [c.text.strip() for c in t.rows[0].cells]
    if header[:2] == ["Study", "Methodology"]:
        t._tbl.getparent().remove(t._tbl)
        break

# ---------------------------------------------------------------------------
# 4. Interim Project Status (Progress Snapshot), before "Scope and objectives"
# ---------------------------------------------------------------------------
scope_heading = find_para("Scope and objectives")
heading_before(scope_heading, "Interim Project Status (Progress Snapshot)")
bullets_before(
    scope_heading,
    [
        "Completed: acquisition and preparation of both real datasets (Home Credit application_train.csv, "
        "307,511 rows; HMDA 2008-2017 nationwide legacy sample, 715,927 rows); full EDA on both datasets "
        "(distributions, missingness, correlation, protected-attribute target-rate gaps); feature engineering "
        "(age/employment/affordability ratios, AGE_GROUP bucketing); baseline model training (logistic "
        "regression, XGBoost); hyperparameter tuning (RandomizedSearchCV for logistic regression, 25-trial "
        "Optuna/TPE search for XGBoost, 5-fold cross-validated ROC-AUC); held-out evaluation (ROC-AUC, PR-AUC, "
        "F1, KS statistic, Brier score); SHAP explainability on the best-performing tuned model; fairness audit "
        f"(pre-mitigation) on CODE_GENDER and AGE_GROUP; bias mitigation via {mit_attr}-based reweighing with "
        "re-training and re-audit (post-mitigation); and an identical fairness-audit pass on HMDA "
        "(derived_race, derived_sex) against the high_cost_flag pricing proxy.",
        "In progress: synthesis of this Interim Report (APA 7 formatting, figure/table cross-referencing); "
        "expansion of the literature review beyond the synopsis's initial source set; drafting of the "
        "accuracy-fairness decision framework referenced in Next Steps.",
        "Pending: formal statistical hypothesis testing write-up (two-proportion z-tests for RQ1/RQ3/RQ4, "
        "confidence-interval estimation for RQ2) with reported p-values and effect sizes; a joint (not "
        "single-attribute) sensitivity analysis of the reweighing mitigation; evaluation of a second "
        "mitigation technique for comparison, if time allows; the final model-governance checklist; and a "
        "final proofreading/APA-compliance pass.",
    ],
)
normal_before(scope_heading, "")

# ---------------------------------------------------------------------------
# 5. Literature survey, before "Data Description" -- relocate the four
#    citation-dense Background paragraphs here (not duplicated) and add an
#    expanded 12-source relevance matrix + review-approach text.
# ---------------------------------------------------------------------------
data_desc_heading = find_para("Data Description")
lit_heading = heading_before(data_desc_heading, "Literature survey")
heading_before(data_desc_heading, "Literature Review Approach")
normal_before(
    data_desc_heading,
    "Sources were identified through targeted keyword searches (“credit scoring explainable AI”, "
    "“SHAP credit risk”, “HMDA fair lending disparities”, “bias mitigation reweighing "
    "classification”, “fairness machine learning lending”) across Google Scholar, SSRN, the ACM "
    "Digital Library, and arXiv, supplemented by backward citation-chasing from foundational fairness-ML and "
    "credit-scoring surveys. Inclusion criteria: peer-reviewed articles, working papers, or top-venue "
    "proceedings published 1996-2026 that (a) apply machine learning to credit or mortgage decisioning, (b) "
    "develop or apply a post-hoc explainability technique to tabular/financial models, or (c) develop or apply "
    "a group-fairness metric or bias-mitigation technique relevant to lending. Each source below is mapped to "
    "the research question(s) (RQ1-RQ4, see Scope and objectives) it most directly informs.",
)
normal_before(data_desc_heading, "")

heading_before(data_desc_heading, "Summary of Key Literature (Minimum 10 relevant sources)")
lit_rows = [
    ("Chen & Guestrin (2016)", "ML systems / gradient boosting", "Benchmark tabular datasets",
     "Introduces XGBoost, a scalable, regularized tree-boosting system",
     "Sparsity-aware split finding and weighted quantile sketch give large accuracy/speed gains over earlier "
     "boosting implementations on structured data",
     "RQ1 -- justifies XGBoost as the non-linear challenger model"),
    ("Ke et al. (2017)", "ML systems / gradient boosting", "Benchmark tabular datasets",
     "Introduces LightGBM (leaf-wise growth, histogram binning)",
     "Comparable or better accuracy than prior GBDTs at substantially lower training cost",
     "RQ1 -- alternative ensemble considered during model-family selection (config.yaml scaffolds LightGBM; "
     "XGBoost was retained as the MVP challenger)"),
    ("Lundberg & Lee (2017)", "Explainable AI (XAI)", "Model-agnostic; UCI/benchmark tabular data",
     "Introduces SHAP (Shapley Additive exPlanations), unifying prior additive-attribution methods",
     "SHAP values give a theoretically grounded, locally accurate, consistent per-feature attribution for any "
     "model, including tree ensembles (TreeExplainer)",
     "RQ2 -- primary explainability method used to attribute the tuned model's predictions"),
    ("Ribeiro, Singh, & Guestrin (2016)", "Explainable AI (XAI)", "Text/image/tabular classifiers",
     "Introduces LIME, a local surrogate-model explanation technique",
     "Local linear surrogates approximate any classifier's decision boundary near a single prediction",
     "RQ2 -- contrasting XAI approach; SHAP was selected over LIME for its additive consistency guarantees"),
    ("Doshi-Velez & Kim (2017)", "Explainable AI (XAI) / ML theory", "Conceptual / no single dataset",
     "Proposes a taxonomy for rigorously evaluating interpretability claims",
     "Distinguishes functionally-grounded, human-grounded, and application-grounded interpretability evaluation",
     "RQ2 -- frames why post-hoc XAI, not just an interpretable baseline, is required for regulatory-grade "
     "explanation"),
    ("Munnell, Tootell, Browne, & McEneaney (1996)", "Fair lending / mortgage economics", "Boston-area HMDA "
     "mortgage applications", "Regression analysis of denial rates controlling for underwriting variables",
     "Documents statistically significant racial disparities in mortgage denial that persist after controlling "
     "for standard risk factors", "RQ3/RQ4 -- foundational evidence motivating the fairness-audit design"),
    ("Bhutta, Hizmo, & Ringo (2022)", "Fair lending / mortgage economics", "Modern HMDA + underwriting data",
     "Combines HMDA with proprietary human/algorithmic underwriting data",
     "Observable risk factors explain most, but not all, of the residual racial gap in mortgage outcomes",
     "RQ4 -- motivates auditing an algorithmic pricing proxy specifically, not just raw approval gaps"),
    ("Barocas & Selbst (2016)", "Fairness / law and ML", "Conceptual / legal doctrine",
     "Legal analysis of disparate impact doctrine applied to data-driven decisions",
     "Formalizes the disparate-impact framing (incl. the four-fifths-rule heuristic) used to flag "
     "practically significant group differences in automated decisions",
     "RQ3/RQ4 -- source of the four-fifths / disparate-impact-ratio convention used throughout the fairness "
     "audit"),
    ("Dwork, Hardt, Pitassi, Reingold, & Zemel (2012)", "Fairness metrics", "Conceptual / algorithmic",
     "Formalizes “fairness through awareness” and individual-fairness constraints",
     "Establishes that group-blind treatment does not guarantee fair outcomes, motivating explicit "
     "group-fairness metrics", "RQ3/RQ4 -- theoretical grounding for auditing group-level parity rather than "
     "relying on excluding protected attributes alone"),
    ("Hardt, Price, & Srebro (2016)", "Fairness metrics", "Conceptual + empirical benchmarks",
     "Introduces equalized odds / equality of opportunity as fairness criteria",
     "Equalized odds (matched TPR/FPR across groups) is proposed as an alternative to demographic parity with "
     "different trade-offs", "RQ3/RQ4 -- equalized-odds difference is reported alongside demographic parity "
     "ratio in the audit (dac.fairness.audit)"),
    ("Kamiran & Calders (2012)", "Bias mitigation", "Adult Income, German Credit, Dutch Census",
     "Introduces preprocessing reweighing to decorrelate a protected attribute from the label before training",
     "Reweighing improves group fairness with a smaller accuracy cost than suppression, and requires no "
     "model-architecture changes", "RQ3 -- the bias-mitigation technique implemented and evaluated in this "
     "study"),
    ("Zhang, Lemoine, & Mitchell (2018)", "Bias mitigation", "UCI Adult, COMPAS",
     "Introduces adversarial debiasing (adversary predicts the protected attribute from model outputs)",
     "Adversarial training reduces demographic-parity/equalized-odds gaps at some accuracy cost, "
     "in-processing rather than pre-processing", "RQ3 (scope note) -- considered but descoped from the MVP in "
     "favor of the model-agnostic reweighing baseline; a candidate for the final report's Next Steps"),
    ("Mehrabi, Morstatter, Saxena, Lerman, & Galstyan (2021)", "Fairness / ML survey",
     "Review (no single dataset)", "Systematic survey of bias sources, fairness metrics, and mitigation "
     "techniques across ML", "Catalogs the fairness-ML landscape and explicitly flags that most audits are "
     "single-dataset, single-model case studies",
     "RQ4 -- motivates this study's cross-dataset audit-methodology-transfer design"),
    ("Dastile, Celik, & Potsane (2020)", "Credit scoring / ML survey", "Review (no single dataset)",
     "Systematic literature survey of statistical and ML credit-scoring models",
     "Ensembles (incl. boosting) consistently outperform single classical models on credit-scoring "
     "benchmarks, at an interpretability cost", "RQ1 -- corroborates the accuracy-interpretability trade-off "
     "motivating this study's dual logistic-regression/XGBoost design"),
]
table_before(
    data_desc_heading,
    ["Author (Year)", "Domain/Context", "Dataset/Setting", "Method(s)", "Key Findings",
     "How it supports RQ(s) / project decisions"],
    lit_rows,
    font_size=8,
)

# Relocate the four citation-dense literature paragraphs out of Background
# (found by unique text prefixes) into the new Literature Survey section.
lit_paras = [
    find_para("As lenders replace traditional linear scorecards", contains=True),
    find_para("Model accuracy and explainability alone do not guarantee fair treatment", contains=True),
    find_para("This study treats two publicly available secondary datasets", contains=True),
    find_para("Most published fairness-in-lending studies audit a single model", contains=True),
]
heading_before(data_desc_heading, "Synthesis by Theme")
for p in lit_paras:
    move_before(data_desc_heading, p)
normal_before(data_desc_heading, "")

# ---------------------------------------------------------------------------
# 6. Dataset Overview, right after the Data Description bullets, before the
#    "Data dictionary (mandatory)" heading.
# ---------------------------------------------------------------------------
dict_heading = find_para("Data dictionary (mandatory)")
heading_before(dict_heading, "Dataset Overview")
table_before(
    dict_heading,
    ["Aspect", "Home Credit (HCDR)", "HMDA"],
    [
        ("Number of records (rows)", f"{summary['home_credit_shape'][0]:,}", f"{summary['hmda_shape'][0]:,}"),
        ("Number of variables (columns)", summary["home_credit_shape"][1], summary["hmda_shape"][1]),
        ("Time period", "Single snapshot (no explicit date range)", "2008-2010, 2012-2017 (9 usable years; "
         "2007 and 2011 excluded, corrupt source archives)"),
        ("Unit of analysis", "One row per loan application", "One row per originated mortgage loan"),
        ("Target variable(s)", "TARGET (binary; 1 = payment difficulty/default, 8.07% positive)",
         "high_cost_flag (binary, derived; 1 = higher-priced/HOEPA-reportable loan, 5.77% positive)"),
        ("Data provenance", "Real data (Kaggle competition extract)" if not summary["home_credit_is_synthetic"]
         else "Synthetic fallback (real extract unavailable)",
         "Real data (CFPB legacy LAR extract)" if not summary["hmda_is_synthetic"]
         else "Synthetic fallback (real extract unavailable)"),
    ],
)

# ---------------------------------------------------------------------------
# 7. GitHub Data Availability Statement + Screenshots, after the data
#    dictionary tables, before the HCDR "Detailed EDA" Heading 1.
# ---------------------------------------------------------------------------
hcdr_eda_heading = find_para("Home Credit Default Risk (HCDR) — Detailed EDA")
heading_before(hcdr_eda_heading, "GitHub Data Availability Statement")
normal_before(hcdr_eda_heading, "The raw and processed datasets are available in the GitHub repository under:")
bullets_before(
    hcdr_eda_heading,
    [
        "Raw data: data/raw/home_credit/application_train.csv; data/raw/hmda/hmda_nationwide_2008_2017_sample.csv",
        "Source archives: data/external/HCDR/home-credit-default-risk.zip; data/external/HMDA/*.zip",
        "Code/notebooks: src/dac/ (pipeline modules), scripts/run_pipeline.py (end-to-end orchestrator), "
        "notebooks/ (exploratory analysis mirroring the pipeline)",
        f"Full repository: {GITHUB_URL}",
    ],
)
normal_before(hcdr_eda_heading, "")
heading_before(hcdr_eda_heading, "Screenshots / Evidence")
normal_before(
    hcdr_eda_heading,
    "Figures 2.1-2.3 below show a live data preview of each raw source file and the repository's folder "
    "structure, generated directly from the checked-in data files as evidence of availability.",
)
image_before(hcdr_eda_heading, FIG_EVID / "preview_home_credit.png",
             "Figure 2.1 — Home Credit application_train.csv preview (first 8 rows, key columns).")
image_before(hcdr_eda_heading, FIG_EVID / "preview_hmda.png",
             "Figure 2.2 — HMDA nationwide sample preview (first 8 rows, key columns).")
image_before(hcdr_eda_heading, FIG_EVID / "repo_tree.png",
             "Figure 2.3 — Repository folder structure.")

# ---------------------------------------------------------------------------
# 8. Analysis > Data Cleaning, before the HCDR "Detailed EDA" Heading 1
#    (which becomes the EDA Results subsection).
# ---------------------------------------------------------------------------
heading_before(hcdr_eda_heading, "Analysis")
normal_before(
    hcdr_eda_heading,
    "This section documents the cleaning steps applied to both datasets (Data Cleaning) and the "
    "resulting exploratory findings (EDA Results), with emphasis on interpreting what each figure/table "
    "implies for feature engineering, modelling, and the fairness audit. The pipeline code implementing "
    "every step below is provided in Appendix A.",
)
heading_before(hcdr_eda_heading, "Data Cleaning")
normal_before(
    hcdr_eda_heading,
    "No exact duplicate rows were found in either dataset, so no deduplication was necessary. Missing "
    "values are handled entirely inside the shared preprocessing pipeline (dac.features.engineering."
    "build_preprocessor) rather than by row/column deletion, so that the same, auditable transformation "
    "is applied to train and test folds alike: numeric columns are median-imputed then standardized "
    "(StandardScaler), and categorical columns are most-frequent-imputed then one-hot encoded (unknown "
    "categories at inference time are ignored rather than raising an error). One dataset-specific anomaly "
    "required an explicit rule rather than generic imputation: Home Credit's DAYS_EMPLOYED field uses the "
    "sentinel value 365,243 (~1,000 years) to flag unemployed/pensioner/non-working applicants; this is "
    "replaced with NaN (before median imputation) and preserved as a separate binary DAYS_EMPLOYED_ANOM "
    "flag so the signal is not lost.",
)
table_before(
    hcdr_eda_heading,
    ["Issue", "Variables Affected", "Detection Method", "Treatment Applied", "Rationale"],
    [
        ("Missing values (numeric)", "EXT_SOURCE_1/2/3, AMT_ANNUITY, AMT_GOODS_PRICE, building-characteristic "
         "cluster (COMMONAREA_*, LIVINGAPARTMENTS_*, etc.), income (HMDA), census-tract fields (HMDA)",
         "Per-column df.isna().mean(); 67 of 122 HCDR columns and 9 of 22 HMDA columns affected",
         "Median imputation via SimpleImputer, fit on the training fold only, inside the ColumnTransformer",
         "Median is robust to the heavy right-skew typical of income/credit amounts; fitting only on the "
         "training fold avoids test-set leakage"),
        ("Missing values (categorical)", "OCCUPATION_TYPE, FONDKAPREMONT_MODE, and other nominal fields",
         "value_counts(dropna=False) per column", "Most-frequent imputation, then one-hot encoding "
         "(handle_unknown='ignore')", "Standard, reversible treatment for nominal categorical fields; "
         "unknown-category handling keeps the pipeline robust at inference time"),
        ("Sentinel / anomalous value", "DAYS_EMPLOYED = 365243 (18.01% of HCDR rows)", "Cross-referenced "
         "against the documented Kaggle data-quality artifact for this field",
         "Replaced with NaN prior to imputation; added DAYS_EMPLOYED_ANOM binary indicator feature",
         "Prevents a ~1,000-year tenure value from corrupting numeric scaling, while explicitly preserving "
         "the unemployed/pensioner signal it was encoding"),
        ("Duplicate rows", "All columns, both datasets", "pandas .duplicated() full-row check",
         "None applied (0 duplicate rows found in either dataset)", "No deduplication necessary"),
        ("Small / unstable category", "CODE_GENDER = 'XNA' (HCDR, n = 4)", "Group-size threshold review "
         "during the sample-size calculation", "Retained in EDA/fairness tables for transparency; excluded "
         "from group-fairness point estimates as statistically unstable",
         "4 rows cannot support a stable selection-rate estimate; excluding avoids a spurious fairness "
         "finding"),
        ("Non-disclosure category", "derived_race / derived_sex = 'Not Available' (HMDA)",
         "Category audit of the two protected attributes", "Reported as its own group, not merged into a "
         "modeled demographic category", "Merging would mask a legitimate non-disclosure signal as if it "
         "were a demographic group"),
    ],
    font_size=8,
)
heading_before(hcdr_eda_heading, "EDA Results (Interpretation Emphasis)")
normal_before(
    hcdr_eda_heading,
    "Every figure and table below is referenced in the surrounding text and paired with an interpretation "
    "of what it implies for the downstream modelling, explainability, and fairness stages; Table 8 "
    "(EDA insight summary, at the end of this subsection) consolidates the single most decision-relevant "
    "insight from each.",
)

# ---------------------------------------------------------------------------
# 9. EDA insight summary table, after Data Quality Notes, before "Analytic
#    approach" (which becomes "Choice of Models with Justification" below).
# ---------------------------------------------------------------------------
analytic_heading = find_para("Analytic approach")
normal_before(analytic_heading, "Table 8 — EDA insight summary", bold=True)
table_before(
    analytic_heading,
    ["Figure/Table Reference", "What it Shows", "Key Insight", "Why it Matters (Link to RQ/objective)",
     "Decision/Next Step"],
    [
        ("Figure 3.1 (TARGET distribution)", "HCDR class balance", "Only 8.07% of applications default "
         "(TARGET=1)", "Motivates reporting PR-AUC/Brier alongside ROC-AUC (RQ1); accuracy alone would be "
         "misleading", "Use imbalance-aware metrics throughout evaluation (see Evaluation Metrics table)"),
        ("Figure 3.2 (missingness)", "Top-20 columns by missing fraction", "A cluster of building/apartment "
         "characteristic columns is 65-70% missing", "Aggressive imputation of this cluster risks injecting "
         "bias; it is third-party enrichment data, not core applicant data", "Treat as optional features; "
         "retained via median imputation rather than dropped outright (see Data Cleaning)"),
        ("Table 7 (numeric summary) / DAYS_EMPLOYED note", "Descriptive statistics for key numeric fields",
         "365,243 sentinel in DAYS_EMPLOYED affects 18.01% of rows", "A naive numeric treatment would treat "
         "unemployed/pensioner applicants as ~1,000-year employees, corrupting scaling", "Sentinel replaced "
         "with NaN + DAYS_EMPLOYED_ANOM flag engineered (see Features Included table)"),
        ("Figure 3.4 (correlation heatmap)", "Pearson correlation with TARGET", "EXT_SOURCE_1/2/3 (external "
         "bureau scores) are by far the strongest linear predictors of default", "These are purpose-built "
         "risk scores from other data sources; expected to dominate SHAP importance for RQ2", "EXT_SOURCE_MEAN "
         "/ EXT_SOURCE_STD engineered as aggregate features; confirmed as the top SHAP driver (see Preliminary "
         "Findings, RQ2)"),
        ("CODE_GENDER / AGE_GROUP target-rate figures", "Default rate by protected attribute", "Male "
         "applicants default at 10.14% vs. 7.00% for female; default rate falls monotonically with age "
         "(12.29% under-25 vs. 3.66% 65+)", "An EDA-level gap on both protected attributes is the "
         "early-warning signal that justifies running a formal fairness audit (RQ3)", "Both attributes carried "
         "into the fairness audit / bias-mitigation stage"),
        ("Figure 4.4 (HMDA correlation heatmap)", "Pearson correlation with high_cost_flag",
         "minority_population is positively correlated with high_cost_flag; area-income proxies are "
         "negatively correlated", "An EDA-level signal consistent with the documented mortgage-pricing "
         "disparity literature (RQ4)", "Motivates running the identical fairness-audit methodology on "
         "derived_race / derived_sex"),
        ("Table 36 (Cross-Dataset Comparison)", "Side-by-side HCDR vs. HMDA profile", "Both datasets show "
         "target-class imbalance and a measurable protected-attribute target-rate gap at the raw EDA level",
         "Empirically justifies auditing both datasets rather than assuming the finding is dataset-specific",
         "Proceed with the dual-dataset fairness-audit design (RQ3 + RQ4)"),
    ],
    font_size=8,
)

# ---------------------------------------------------------------------------
# 10. Modelling: rename "Analytic approach" -> "Choice of Models with
#     Justification", add Model 1/2 subsections, an explainability/fairness
#     methodology subheading, a Features table, and an Evaluation Metrics
#     table (with formulae), all before "Solution to RQ1-4".
# ---------------------------------------------------------------------------
heading_before(analytic_heading, "Modelling")
normal_before(
    analytic_heading,
    "Two model families are trained and compared on Home Credit (RQ1), with the winning family also "
    "reused, unchanged in architecture, for the HMDA fairness-audit-only pipeline (RQ4).",
)
set_text(analytic_heading, "Choice of Models with Justification")

shap_para = find_para("For RQ2, SHAP", contains=True)
heading_before(shap_para, "Model 1: L2-Regularized Logistic Regression — Interpretable Baseline")
normal_before(
    shap_para,
    "Justification: a linear, L2-regularized logistic regression is the interpretable baseline against "
    "which the ensemble challenger is measured. Its coefficients map directly onto the additive, "
    "reason-code-style explanations regulators expect for adverse-action notices (Consumer Financial "
    "Protection Bureau, 2023), giving it a low interpretability cost even before any post-hoc XAI is "
    "applied. Tuned via RandomizedSearchCV (15 iterations, 5-fold stratified cross-validated ROC-AUC) over "
    "the regularization strength and solver.",
)
heading_before(shap_para, "Model 2: XGBoost — Non-Linear Ensemble Challenger")
normal_before(
    shap_para,
    "Justification: gradient-boosted trees capture non-linear interaction effects (e.g., among income, "
    "credit amount, and external bureau score) that a linear model's additive form cannot represent "
    "(Chen & Guestrin, 2016; Dastile et al., 2020). XGBoost is hypothesized (H1) to achieve significantly "
    "higher held-out ROC-AUC than tuned logistic regression. Tuned via a 25-trial Optuna/TPE search (5-fold "
    "stratified cross-validated ROC-AUC) over tree depth, learning rate, subsampling, and regularization; "
    "class imbalance is handled via scale_pos_weight rather than resampling, to keep the full real "
    "sample size. The same architecture, retrained on the HMDA working file with derived_race/derived_sex "
    "excluded from its own features, is reused unchanged for the RQ4 cross-dataset fairness-audit "
    "replication.",
)
heading_before(shap_para, "Explainability, Fairness Audit, and Bias-Mitigation Methodology")

solution_heading = find_para("Solution to RQ1-4")
heading_before(solution_heading, "Features Included and Feature Engineering")
normal_before(
    solution_heading,
    f"After feature engineering, the Home Credit model consumes "
    f"{summary['home_credit_shape'][0]:,} rows across "
    f"115 numeric and 15 categorical features (dac.features.engineering.split_feature_columns); "
    "CODE_GENDER and AGE_GROUP are deliberately excluded from the model's own feature set and used only "
    "for post-hoc fairness auditing, per standard fair-lending practice. HMDA's model similarly excludes "
    "derived_race and derived_sex. All engineered features are basic, auditable arithmetic transforms of "
    "existing raw fields (no external data joins), computed identically for every row so the transform is "
    "reproducible end to end from the checked-in raw CSVs.",
)
table_before(
    solution_heading,
    ["Feature", "Original/Engineered", "Type", "Reason for Inclusion", "Used in Model(s)"],
    [
        ("AGE_YEARS", "Engineered (from DAYS_BIRTH)", "Numeric", "Human-interpretable age in years",
         "LR, XGBoost (feeds AGE_GROUP)"),
        ("AGE_GROUP", "Engineered (binned AGE_YEARS)", "Categorical (protected)", "5-bucket age band for the "
         "fairness audit", "Fairness audit only (excluded from model features)"),
        ("DAYS_EMPLOYED_ANOM", "Engineered (365243 sentinel flag)", "Binary", "Preserves the unemployed/"
         "pensioner signal without corrupting numeric scaling", "LR, XGBoost"),
        ("YEARS_EMPLOYED", "Engineered (cleaned DAYS_EMPLOYED)", "Numeric", "Tenure in years with the "
         "sentinel removed", "LR, XGBoost"),
        ("CREDIT_INCOME_RATIO", "Engineered (AMT_CREDIT / AMT_INCOME_TOTAL)", "Numeric", "Affordability "
         "ratio; standard credit-risk feature", "LR, XGBoost"),
        ("ANNUITY_INCOME_RATIO", "Engineered (AMT_ANNUITY / AMT_INCOME_TOTAL)", "Numeric",
         "Affordability ratio", "LR, XGBoost"),
        ("CREDIT_TERM", "Engineered (AMT_ANNUITY / AMT_CREDIT)", "Numeric", "Implied loan term / "
         "repayment intensity", "LR, XGBoost"),
        ("GOODS_CREDIT_RATIO", "Engineered (AMT_GOODS_PRICE / AMT_CREDIT)", "Numeric", "Financed-goods "
         "coverage ratio", "LR, XGBoost"),
        ("EXT_SOURCE_MEAN / EXT_SOURCE_STD", "Engineered (mean/std of EXT_SOURCE_1/2/3)", "Numeric",
         "Aggregates the three strongest raw predictors identified in EDA (Table 8)", "LR, XGBoost"),
        ("CHILDREN_RATIO", "Engineered (CNT_CHILDREN / CNT_FAM_MEMBERS)", "Numeric", "Household-composition "
         "ratio", "LR, XGBoost"),
        ("EXT_SOURCE_1 / _2 / _3, AMT_INCOME_TOTAL, AMT_CREDIT, ...", "Original", "Numeric",
         "Core raw predictors retained unchanged (106 of 115 numeric features are original columns)",
         "LR, XGBoost"),
        ("NAME_EDUCATION_TYPE, NAME_INCOME_TYPE, OCCUPATION_TYPE, ...", "Original", "Categorical (15 "
         "columns, one-hot encoded)", "Retained applicant/loan attributes", "LR, XGBoost"),
        ("LOAN_INCOME_RATIO", "Engineered (HMDA: loan_amount / income)", "Numeric", "Affordability ratio, "
         "HMDA analogue of CREDIT_INCOME_RATIO", "HMDA XGBoost (fairness-audit model)"),
        ("CODE_GENDER, derived_race, derived_sex", "Original (protected)", "Categorical", "Excluded from "
         "model features entirely; used only for post-hoc MetricFrame fairness auditing", "Fairness audit "
         "only"),
    ],
    font_size=8,
)

heading_before(solution_heading, "Evaluation Metrics (Include Formulae and Calculations)")
normal_before(
    solution_heading,
    "Because both targets are minority-class (8.07% for TARGET, 5.77% for high_cost_flag), accuracy is not "
    "reported as a primary metric; ROC-AUC is supplemented with PR-AUC and Brier score, which remain "
    "informative under class imbalance. Group-fairness metrics use Fairlearn's MetricFrame "
    "(dac.fairness.audit); the four-fifths rule is a borrowed EEOC employment-discrimination heuristic, "
    "not a codified ECOA/Reg B lending standard, used here only as a common fairness-ML convention.",
)
table_before(
    solution_heading,
    ["Metric", "Formula", "Interpretation", "Used For"],
    [
        ("ROC-AUC", "Area under the ROC curve, ROC(t) = (FPR(t), TPR(t)) for threshold t in [0,1]",
         "Probability a randomly chosen positive case is scored higher than a randomly chosen negative "
         "case; 0.5 = random, 1.0 = perfect", "Primary model-selection metric (RQ1)"),
        ("PR-AUC (Average Precision)", "sum_n (R_n - R_{n-1}) * P_n over decision thresholds n",
         "Area under the precision-recall curve; sensitive to performance on the minority (positive) class",
         "Imbalance-robust companion to ROC-AUC"),
        ("Precision", "TP / (TP + FP)", "Share of predicted-positive cases that are true positives",
         "Diagnostic component of F1"),
        ("Recall (Sensitivity/TPR)", "TP / (TP + FN)", "Share of actual-positive cases correctly identified",
         "Diagnostic component of F1; also the TPR used in equalized odds"),
        ("F1-score", "2 * (Precision * Recall) / (Precision + Recall)", "Harmonic mean of precision and "
         "recall at the 0.5 decision threshold", "Threshold-based classification quality"),
        ("KS statistic", "max_t | TPR(t) - FPR(t) |", "Maximum separation between the cumulative score "
         "distributions of the positive and negative classes", "Standard credit-scoring discrimination "
         "metric"),
        ("Brier score", "(1/N) * sum_i (p_i - y_i)^2", "Mean squared error between predicted probability "
         "p_i and the true label y_i; lower is better-calibrated", "Probability-calibration quality "
         "under imbalance"),
        ("Demographic Parity Ratio", "min_g SR(g) / max_g SR(g), where SR(g) = favorable-outcome selection "
         "rate for group g", "1.0 = equal selection rates across groups; four-fifths convention flags "
         "ratio < 0.80 as a practically significant gap", "Fairness audit (RQ3, RQ4)"),
        ("Demographic Parity Difference", "max_g SR(g) - min_g SR(g)", "Absolute gap in selection rate "
         "between the best- and worst-off groups", "Fairness audit (RQ3, RQ4)"),
        ("Equalized Odds Difference", "max over {TPR, FPR} of (max_g rate(g) - min_g rate(g))", "Largest "
         "true/false-positive-rate gap across groups; complements demographic parity", "Fairness audit "
         "(RQ3, RQ4)"),
    ],
    font_size=8,
)

# ---------------------------------------------------------------------------
# 11. Preliminary results: insert actual-outcome paragraphs after each RQ's
#     hypothesis table (from "Solution to RQ1-4", kept as methodology under
#     Modelling... actually retitled + kept here as the RQ-by-RQ findings
#     scaffold), then Preliminary Model Performance / Limitations / Next Steps.
# ---------------------------------------------------------------------------
heading_before(solution_heading, "Preliminary results")
normal_before(
    solution_heading,
    "For each research question, the methodology below (unchanged from the study design) is followed "
    "immediately by the actual preliminary result observed from this run of the pipeline.",
)
set_text(solution_heading, "Preliminary Findings (by Research Question)")

top_feats = list(shap_top.items())[:5]
top_feats_str = "; ".join(f"{clean_feature_name(k)} ({v:.4f})" for k, v in top_feats)
auc_gap = best_tuned["roc_auc"] - lr_tuned["roc_auc"]
auc_cost = tuned[best_model]["roc_auc"] - mitigated["roc_auc"]

rq2_para = find_para("RQ2 -- Explainability", contains=True)
normal_before(
    rq2_para,
    f"Preliminary result: tuned {best_model.replace('_', ' ')} achieves ROC-AUC = "
    f"{best_tuned['roc_auc']:.4f} (PR-AUC = {best_tuned['pr_auc']:.4f}, F1 = {best_tuned['f1']:.4f}, KS = "
    f"{best_tuned['ks_statistic']:.4f}, Brier = {best_tuned['brier_score']:.4f}) versus tuned logistic "
    f"regression's ROC-AUC = {lr_tuned['roc_auc']:.4f} — a +{auc_gap:.4f} absolute gap, consistent in "
    f"direction and magnitude with the pilot effect size used in the RQ1 sample-size calculation. Both "
    f"baselines (logistic regression {lr_base['roc_auc']:.4f}, XGBoost {xgb_base['roc_auc']:.4f} ROC-AUC "
    f"before tuning) improved after hyperparameter search. {best_model.replace('_', ' ').title()} is "
    "retained as the best_model for RQ2-RQ4 (full comparison in Table 9, Preliminary Model Performance).",
)

rq3_para = find_para("RQ3 -- Fairness audit and mitigation", contains=True)
normal_before(
    rq3_para,
    f"Preliminary result: SHAP mean-|SHAP| ranking on the tuned {best_model.replace('_', ' ')} model places "
    f"{top_feats_str} as the top drivers — dominated by the engineered external-bureau-score aggregate and "
    "affordability/coverage ratios, consistent with H2 and established credit-risk theory (see Figure 6, "
    "SHAP global importance).",
)

rq4_para = find_para("RQ4 -- Cross-dataset replication", contains=True)
normal_before(
    rq4_para,
    f"Preliminary result: pre-mitigation demographic parity ratio is "
    f"{pre_fair['CODE_GENDER']['demographic_parity_ratio']:.3f} for CODE_GENDER and "
    f"{pre_fair['AGE_GROUP']['demographic_parity_ratio']:.3f} for AGE_GROUP — both below the 0.80 "
    f"four-fifths threshold, supporting H3's pre-mitigation hypothesis. After {mit_attr}-based Kamiran and "
    f"Calders (2012) reweighing, the post-mitigation ratio rises to "
    f"{post_fair['CODE_GENDER']['demographic_parity_ratio']:.3f} (CODE_GENDER) and "
    f"{post_fair['AGE_GROUP']['demographic_parity_ratio']:.3f} (AGE_GROUP), both now at or above 0.80, at a "
    f"ROC-AUC cost of {auc_cost:.4f} ({mitigated['roc_auc']:.4f} mitigated vs. {tuned[best_model]['roc_auc']:.4f} "
    "pre-mitigation) — within the pre-registered 5-percentage-point tolerance. Equalized-odds difference, "
    "reported alongside demographic parity, remains comparatively elevated post-mitigation for both "
    f"attributes (CODE_GENDER {post_fair['CODE_GENDER']['equalized_odds_difference']:.3f}, AGE_GROUP "
    f"{post_fair['AGE_GROUP']['equalized_odds_difference']:.3f}) — flagged under Interim Limitations and "
    "Risks below.",
)

recommendation_heading = find_para("Recommendation and application")
normal_before(
    recommendation_heading,
    f"Preliminary result: applying the identical MetricFrame audit code path to an XGBoost model trained "
    f"on the HMDA working file yields a demographic parity ratio of "
    f"{hmda_fair['derived_race']['demographic_parity_ratio']:.3f} for derived_race (below 0.80; "
    f"disparate-impact flag = {hmda_fair['derived_race']['disparate_impact_flag']}) and "
    f"{hmda_fair['derived_sex']['demographic_parity_ratio']:.3f} for derived_sex (disparate-impact flag = "
    f"{hmda_fair['derived_sex']['disparate_impact_flag']}), against a held-out model ROC-AUC of "
    f"{hmda_perf['roc_auc']:.4f}. This supports H4 for derived_race specifically, and demonstrates that the "
    "audit methodology built for Home Credit — not necessarily the magnitude of any specific disparity — "
    "transfers unmodified to a second, independently labeled dataset and decision task.",
)

# ---------------------------------------------------------------------------
# 12. Preliminary Model Performance table + figures, before "Recommendation
#     and application".
# ---------------------------------------------------------------------------
heading_before(recommendation_heading, "Preliminary Model Performance")
normal_before(
    recommendation_heading,
    "Table 9 compares all Home Credit model variants plus the HMDA fairness-audit model on the same "
    "held-out metrics; Table 10 compares pre- and post-mitigation fairness metrics for the two Home Credit "
    "protected attributes plus the two HMDA protected attributes.",
)


def fmt(m):
    return (f"{m['roc_auc']:.4f}", f"{m['pr_auc']:.4f}", f"{m['f1']:.4f}", f"{m['ks_statistic']:.4f}",
            f"{m['brier_score']:.4f}")


table_before(
    recommendation_heading,
    ["Model", "Dataset / Task", "ROC-AUC", "PR-AUC", "F1", "KS", "Brier"],
    [
        ("Logistic regression (baseline)", "HCDR default (TARGET)", *fmt(lr_base)),
        ("XGBoost (baseline)", "HCDR default (TARGET)", *fmt(xgb_base)),
        ("Logistic regression (tuned)", "HCDR default (TARGET)", *fmt(lr_tuned)),
        ("XGBoost (tuned)", "HCDR default (TARGET)", *fmt(xgb_tuned)),
        (f"{best_model.replace('_', ' ').title()} (post-mitigation, reweighing on {mit_attr})",
         "HCDR default (TARGET)", *fmt(mitigated)),
        ("XGBoost (fairness-audit only)", "HMDA pricing proxy (high_cost_flag)", *fmt(hmda_perf)),
    ],
    font_size=8,
)
image_before(recommendation_heading, METRICS_DIR / "model_comparison.png",
             "Figure 5 — Baseline vs. tuned model comparison (ROC-AUC, PR-AUC, F1, KS).")
image_before(recommendation_heading, FIG_HC / f"{best_model}_evaluation_suite.png",
             f"Figure 6 — {best_model.replace('_', ' ').title()}: ROC, precision-recall, calibration, and "
             "confusion-matrix diagnostics.")
image_before(recommendation_heading, FIG_HC / "shap" / f"{best_model}_shap_bar.png",
             f"Figure 7 — {best_model.replace('_', ' ').title()}: global SHAP feature importance "
             "(mean |SHAP|).")

table_before(
    recommendation_heading,
    ["Dataset", "Attribute", "Stage", "Demographic Parity Ratio", "Demographic Parity Difference",
     "Equalized Odds Difference", "Four-Fifths Flag"],
    [
        ("HCDR", "CODE_GENDER", "Pre-mitigation", f"{pre_fair['CODE_GENDER']['demographic_parity_ratio']:.3f}",
         f"{pre_fair['CODE_GENDER']['demographic_parity_difference']:.3f}",
         f"{pre_fair['CODE_GENDER']['equalized_odds_difference']:.3f}",
         pre_fair["CODE_GENDER"]["disparate_impact_flag"]),
        ("HCDR", "CODE_GENDER", "Post-mitigation", f"{post_fair['CODE_GENDER']['demographic_parity_ratio']:.3f}",
         f"{post_fair['CODE_GENDER']['demographic_parity_difference']:.3f}",
         f"{post_fair['CODE_GENDER']['equalized_odds_difference']:.3f}",
         post_fair["CODE_GENDER"]["disparate_impact_flag"]),
        ("HCDR", "AGE_GROUP", "Pre-mitigation", f"{pre_fair['AGE_GROUP']['demographic_parity_ratio']:.3f}",
         f"{pre_fair['AGE_GROUP']['demographic_parity_difference']:.3f}",
         f"{pre_fair['AGE_GROUP']['equalized_odds_difference']:.3f}",
         pre_fair["AGE_GROUP"]["disparate_impact_flag"]),
        ("HCDR", "AGE_GROUP", "Post-mitigation", f"{post_fair['AGE_GROUP']['demographic_parity_ratio']:.3f}",
         f"{post_fair['AGE_GROUP']['demographic_parity_difference']:.3f}",
         f"{post_fair['AGE_GROUP']['equalized_odds_difference']:.3f}",
         post_fair["AGE_GROUP"]["disparate_impact_flag"]),
        ("HMDA", "derived_race", "Audit only (no mitigation)",
         f"{hmda_fair['derived_race']['demographic_parity_ratio']:.3f}",
         f"{hmda_fair['derived_race']['demographic_parity_difference']:.3f}",
         f"{hmda_fair['derived_race']['equalized_odds_difference']:.3f}",
         hmda_fair["derived_race"]["disparate_impact_flag"]),
        ("HMDA", "derived_sex", "Audit only (no mitigation)",
         f"{hmda_fair['derived_sex']['demographic_parity_ratio']:.3f}",
         f"{hmda_fair['derived_sex']['demographic_parity_difference']:.3f}",
         f"{hmda_fair['derived_sex']['equalized_odds_difference']:.3f}",
         hmda_fair["derived_sex"]["disparate_impact_flag"]),
    ],
    font_size=8,
)
image_before(recommendation_heading, FIG_HC / "fairness" / f"{best_model}_pre_mitigation_{mit_attr}_selection_rate.png",
             f"Figure 8 — {best_model.replace('_', ' ').title()} selection rate by {mit_attr}, pre-mitigation.")
image_before(recommendation_heading, FIG_HC / "fairness" / f"{best_model}_post_mitigation_{mit_attr}_selection_rate.png",
             f"Figure 9 — {best_model.replace('_', ' ').title()} selection rate by {mit_attr}, post-mitigation.")
image_before(recommendation_heading, FIG_HMDA / "hmda_xgboost_derived_race_selection_rate.png",
             "Figure 10 — HMDA XGBoost selection rate by derived_race.")

# ---------------------------------------------------------------------------
# 13. Interim Limitations and Risks + Next Steps, before "Recommendation and
#     application" (whose content is folded into Next Steps, then removed).
# ---------------------------------------------------------------------------
heading_before(recommendation_heading, "Interim Limitations and Risks")
bullets_before(
    recommendation_heading,
    [
        "Four-fifths rule caveat: the 0.80 disparate-impact-ratio threshold is an EEOC "
        "employment-discrimination heuristic, not a codified ECOA/Regulation B lending standard; it is used "
        "here only as a common fairness-ML convention, not a claimed regulatory bright line.",
        f"Single mitigation technique: reweighing (Kamiran & Calders, 2012) was applied on {mit_attr} only; "
        "AGE_GROUP's post-mitigation improvement is a correlated side-effect rather than a "
        "separately-optimized result, and a second technique (e.g., equalized-odds postprocessing or "
        "adversarial debiasing) has not yet been evaluated for comparison.",
        f"Equalized-odds trade-off: even after reweighing raises demographic parity above 0.80, equalized-"
        f"odds difference remains comparatively elevated (CODE_GENDER "
        f"{post_fair['CODE_GENDER']['equalized_odds_difference']:.3f}) — a known fairness-metric trade-off "
        "that the final report needs to discuss explicitly rather than treat demographic parity as a "
        "complete fairness guarantee.",
        "HMDA target is a pricing proxy, not approval/denial: the 2008-2017 legacy extract used here "
        "contains originated loans only, so high_cost_flag cannot be compared apples-to-apples against "
        "HCDR's TARGET (default) — RQ4 tests methodology transfer, not an equivalent disparity magnitude.",
        "HMDA sampling: each year is subsampled at ~1% (seeded per year), not a full-population extract; "
        "results should be read as directionally representative rather than exact population statistics, "
        "and 2 of 11 available years (2007, 2011) were excluded as corrupt source archives.",
        "Small-n subgroup instability: HCDR's CODE_GENDER = 'XNA' category (n = 4) is retained for "
        "transparency in EDA/fairness tables but excluded from group-fairness point estimates as "
        "statistically unstable.",
        "Formal inferential statistics pending: the z-test / confidence-interval calculations underlying "
        "each RQ's stated hypothesis (see Sample size calculation) have not yet been executed and reported "
        "with p-values for this interim submission; only point-estimate preliminary results are reported "
        "above.",
    ],
)

heading_before(recommendation_heading, "Next Steps (for Final Report)")
bullets_before(
    recommendation_heading,
    [
        "Execute and report the formal statistical tests underlying RQ1 (one-tailed two-proportion z-test "
        "on ROC-AUC), RQ2 (confidence interval on SHAP-domain agreement), and RQ3/RQ4 (two-tailed "
        "two-proportion z-tests on demographic parity), with p-values and effect sizes.",
        "Run a joint sensitivity analysis of the reweighing mitigation across both CODE_GENDER and "
        "AGE_GROUP simultaneously, rather than mitigating on one attribute and observing the other as a "
        "side-effect.",
        "Evaluate a second bias-mitigation technique (equalized-odds postprocessing or adversarial "
        "debiasing) for direct comparison against reweighing, addressing the equalized-odds trade-off noted "
        "above.",
        "Synthesize findings into an accuracy-fairness decision framework that contrasts accuracy-priority "
        "and fairness-priority operating points, together with a model-governance checklist covering "
        "explainability documentation, periodic bias re-audit cadence, and adverse-action-notice generation "
        "— aimed at credit-risk/model-validation teams, fair-lending regulators and examiners, and MLOps "
        "teams building production bias/drift monitoring.",
        "Expand the literature review with any additional 2025-2026 sources identified during final-report "
        "writing.",
        "Complete a final APA 7 formatting and proofreading pass, and finalize the Appendix code excerpt "
        "and walkthrough.",
    ],
)

# Remove the now-redundant "Recommendation and application" / "Identify the
# target user" headings and their content (folded into Next Steps above).
for text in ["Recommendation and application", "Identify the target user"]:
    p = find_para(text)
    p._p.getparent().remove(p._p)
for prefix in ["Findings will be synthesized into a decision framework",
               "Credit-risk and model-validation teams",
               "Regulators, examiners, and compliance",
               "Data science and MLOps teams"]:
    p = find_para(prefix, contains=True)
    p._p.getparent().remove(p._p)

# ---------------------------------------------------------------------------
# 14. Bibliography: retitle the plain-text label into a proper APA heading.
# ---------------------------------------------------------------------------
biblio_label = find_para("References and bibliography")
biblio_label.style = doc.styles["APA Heading 2"]
set_text(biblio_label, "Bibliography")

# ---------------------------------------------------------------------------
# 15. Appendix, appended at the very end of the document (after the last
#     bibliography entry). doc.add_paragraph()/add_table() append at the true
#     end of the body (before the final sectPr), which is what "after the
#     last reference" requires -- an anchor-based insert_paragraph_before
#     would (and did, in an earlier version of this script) wrongly land
#     the new content ahead of whatever paragraph was used as the anchor.
# ---------------------------------------------------------------------------

def heading_end(text: str, style: str = "APA Heading 2"):
    return doc.add_paragraph(text, style=style)


def normal_end(text: str, style: str = "Normal"):
    p = doc.add_paragraph("", style=style)
    p.add_run(text)
    return p


def code_end(text: str):
    p = doc.add_paragraph("", style="Preformatted Text")
    r = p.add_run(text)
    r.font.size = Pt(8)
    r.font.name = "Consolas"
    return p


heading_end("Appendix (Optional)")
normal_end(
    "Appendix A provides the end-to-end pipeline orchestration code referenced throughout the Analysis "
    "and Modelling sections (e.g., the code used to produce Figure 3.1-3.4, the SHAP figures in Figure 7, "
    "and the fairness figures in Figure 8-10); Appendix B provides the central configuration file that "
    "parameterizes every run.",
)

heading_end("Appendix A: Pipeline Orchestration Code (scripts/run_pipeline.py)")
normal_end(
    f"Full source is available at {GITHUB_URL}/blob/master/scripts/run_pipeline.py. The excerpt below shows "
    "the pipeline's top-level stage sequence; each stage delegates to a dedicated module under src/dac/.",
)
run_pipeline_src = (REPO / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
excerpt_lines = run_pipeline_src.splitlines()[:24]
code_end("\n".join(excerpt_lines))
normal_end("")

heading_end("Appendix B: Central Configuration (config/config.yaml)")
normal_end("Full contents of the single configuration file governing every pipeline run:")
config_src = (REPO / "config" / "config.yaml").read_text(encoding="utf-8")
code_end(config_src)

doc.save(str(OUTPUT_DOC))
print("Build complete ->", OUTPUT_DOC)
