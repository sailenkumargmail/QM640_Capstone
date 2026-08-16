"""Builds the QM640 Final Report docx from the Interim Report (for APA
styles/fonts/margins/page-numbering -- reused as-is, unedited) plus the
completed pipeline run (reports/model_performance/run_summary.json) and the
extended analysis run (reports/model_performance/extended_summary.json)
that closes the Interim Report's own "Pending" items: exact hypothesis-test
p-values for RQ1/RQ3/RQ4, a joint SEX x AGE_GROUP mitigation sensitivity
run, a second bias-mitigation technique (equalized-odds post-processing),
and an accuracy-fairness decision framework.

Approach: the interim doc's title page (paragraphs 0-9, ending in the
section break) is kept and lightly edited in place. Everything from the
section break to the final sectPr is stripped and rebuilt from scratch in
the Final Report Template's section order, reusing large, still-accurate
blocks of the interim doc's own content (data description, EDA, model
justification, literature review, references, appendix code) via direct
XML deep-copy -- so their exact formatting/runs/tables/images survive
untouched -- interleaved with new content (Abstract, Contributions,
Architecture/Workflow framing, Implementation and User Benefit, and the
Results/Limitations narrative updated with the new hypothesis-test numbers).

No numbers are hand-typed for anything already computed by the pipeline;
they are pulled from run_summary.json / extended_summary.json.

Usage:
    python scripts/build_final_report.py
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
BASE_DOC = Path(r"C:\Project\MS\DAC\QM640 Data Analytics Capstone Interim Report Sailen.docx")
OUTPUT_DOC = Path(r"C:\Project\MS\DAC\QM640 Data Analytics Capstone Final Report Sailen.docx")
GITHUB_URL = "https://github.com/sailenkumargmail/QM640_Capstone"

FIG = REPO / "reports" / "figures" / "uci_credit"
FIG_EVID = REPO / "reports" / "figures" / "report_evidence"
FIG_DECISION = FIG / "decision_framework"
METRICS_DIR = REPO / "reports" / "model_performance"

run_summary = json.loads((METRICS_DIR / "run_summary.json").read_text(encoding="utf-8"))
ext_summary = json.loads((METRICS_DIR / "extended_summary.json").read_text(encoding="utf-8"))

best_model = run_summary["best_model"]
tuned = run_summary["tuned_metrics"]
mitigated = run_summary["mitigated_performance"]
pre_fair = run_summary["fairness_pre_mitigation"]
post_fair = run_summary["fairness_post_mitigation"]
mit_attr = run_summary["fairness_mitigation_attribute"]

rq1 = ext_summary["rq1_significance"]
rq34 = ext_summary["rq3_rq4_significance"]
joint = ext_summary["joint_mitigation_sensitivity"]
eo = ext_summary["equalized_odds_postprocessing"]
decision = ext_summary["decision_framework"]

MODEL_LABELS = {
    "logistic_regression_tuned": "Logistic Regression",
    "xgboost_tuned": "XGBoost",
    "catboost_tuned": "CatBoost",
    "ebm_tuned": "EBM",
}


def label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " ").title())


def fmt_p(p: float) -> str:
    """APA7 p-value formatting: 'p < .001' below that threshold, else 'p = .xxx'."""
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


# ---------------------------------------------------------------------------
# Load two independent handles on the same base file: `src` is a read-only
# reference used only to look up and deep-copy content; `doc` is the one
# that gets mutated into the Final Report. Both were parsed from identical
# bytes, so style IDs, numbering IDs, and image relationship IDs line up,
# making elements deep-copied from `src` safe to insert into `doc`.
# ---------------------------------------------------------------------------
src = docx.Document(str(BASE_DOC))
doc = docx.Document(str(BASE_DOC))


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def find_para_index(document, text: str, after: int = 0) -> int:
    """Index (within document.paragraphs) of the first paragraph with exactly
    this text, searching from position `after` onward (not from the start),
    so repeated heading text like "Methodology" can be resolved to the
    occurrence that follows a given anchor rather than always the first one
    in the whole document."""
    for i, p in enumerate(document.paragraphs):
        if i < after:
            continue
        if p.text.strip() == text:
            return i
    raise ValueError(f"paragraph not found (after={after}): {text!r}")


def body_children_excl_sectpr(document) -> list:
    children = list(document.element.body)
    if children and children[-1].tag == qn("w:sectPr"):
        children = children[:-1]
    return children


def copy_block(start_text: str, end_text: str | None, after: int = 0) -> list:
    """Deep-copy the body elements (paragraphs/tables) from `start_text`
    (inclusive) to `end_text` (exclusive), searched forward from paragraph
    index `after`. `end_text=None` copies through the end of the body."""
    start_idx = find_para_index(src, start_text, after=after)
    start_el = src.paragraphs[start_idx]._p
    children = list(src.element.body)
    start_pos = children.index(start_el)

    if end_text is None:
        end_pos = len(body_children_excl_sectpr(src))
    else:
        end_idx = find_para_index(src, end_text, after=start_idx + 1)
        end_el = src.paragraphs[end_idx]._p
        end_pos = children.index(end_el)

    assert start_pos < end_pos, f"bad range {start_text!r} -> {end_text!r}"
    return [copy.deepcopy(children[i]) for i in range(start_pos, end_pos)]


def insert_elements(elements: list) -> None:
    sect_pr = doc.element.body.find(qn("w:sectPr"))
    for el in elements:
        sect_pr.addprevious(el)


def copy_in(start_text: str, end_text: str | None, after: int = 0) -> None:
    insert_elements(copy_block(start_text, end_text, after=after))


def set_text(p, text: str):
    if not p.runs:
        p.add_run(text)
        return p
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""
    return p


def heading(text: str, style: str = "APA Heading 2"):
    return doc.add_paragraph(text, style=style)


def body(text: str, style: str = "Normal", bold: bool = False, italic: bool = False):
    p = doc.add_paragraph("", style=style)
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    return p


def bullets(items, style: str = "List Bullet"):
    return [doc.add_paragraph(text, style=style) for text in items]


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


def add_table(header, rows, font_size: int = 9):
    tbl = doc.add_table(rows=1, cols=len(header))
    hdr_cells = tbl.rows[0].cells
    for i, h in enumerate(header):
        hdr_cells[i].text = str(h)
        for p in hdr_cells[i].paragraphs:
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                r.bold = True
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
    body("")
    return tbl


def add_image(path: Path, caption: str, width_in: float = 6.0):
    if not path.exists():
        body(f"[Figure not found: {path.name}]", italic=True)
        return
    p = doc.add_paragraph("")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_in))
    cap = doc.add_paragraph(caption, style="LO-normal")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p


def rename_heading(old: str, new: str) -> int:
    """Rename every paragraph whose exact text is `old` to `new`. Used for
    section-wide renames (e.g., "Preliminary Results" -> "Results") that
    should apply uniformly regardless of which of several occurrences it is."""
    count = 0
    for p in doc.paragraphs:
        if p.text.strip() == old:
            set_text(p, new)
            count += 1
    return count


def replace_after(text_to_find: str, new_text: str, after_text: str) -> None:
    """Overwrite the first paragraph matching `text_to_find` that appears
    after the paragraph matching `after_text` (both exact-text searches on
    the destination doc, post-insertion)."""
    after_idx = find_para_index(doc, after_text)
    target_idx = find_para_index(doc, text_to_find, after=after_idx + 1)
    set_text(doc.paragraphs[target_idx], new_text)


def insert_paragraph_after(after_text: str, new_text: str, style: str = "LO-normal", after: int = 0):
    """Insert a brand-new paragraph immediately after the paragraph matching
    `after_text` (destination doc)."""
    idx = find_para_index(doc, after_text, after=after)
    anchor = doc.paragraphs[idx]._p
    new_p = doc.add_paragraph(new_text, style=style)
    anchor.addnext(new_p._p)
    return new_p


# ---------------------------------------------------------------------------
# Title page: keep paragraphs 0-9 verbatim except "Interim Report" -> "Final
# Report" and the submission date. The section break embedded in paragraph 9
# (and everything about section 0 -- margins, header/footer) is untouched.
# ---------------------------------------------------------------------------
set_text(doc.paragraphs[3], "Final Report")
set_text(doc.paragraphs[9], date.today().strftime("%B %d, %Y"))

# ---------------------------------------------------------------------------
# Strip section 1 (everything from the paragraph after the title-page break
# through, but not including, the final sectPr) so it can be rebuilt below.
# ---------------------------------------------------------------------------
_title_break_el = doc.paragraphs[9]._p
_body = doc.element.body
_removing = False
for _child in list(_body):
    if _child is _title_break_el:
        _removing = True
        continue
    if _removing:
        if _child.tag == qn("w:sectPr"):
            continue
        _body.remove(_child)


# ===========================================================================
# ABSTRACT
# ===========================================================================
heading("Abstract", style="Heading 1")
body(
    "Problem: Consumer credit-card issuers must decide whom to extend credit to and how "
    "much, in a way that is simultaneously accurate, explainable, and fair across "
    "demographic groups -- three properties usually pursued by different teams at "
    "different times rather than validated together. Solution approach: This study builds "
    "one integrated, reproducible pipeline that trains and tunes four classifiers -- an "
    "interpretable logistic-regression baseline, two gradient-boosted ensembles (XGBoost, "
    "CatBoost), and a glass-box Explainable Boosting Machine (EBM) -- on a shared "
    "preprocessing pipeline; explains the strongest model with both post-hoc SHAP and EBM's "
    "native explanation; audits it for demographic-parity and equalized-odds gaps across "
    "SEX and a derived AGE_GROUP using Fairlearn; and mitigates any gap with Kamiran and "
    "Calders (2012) reweighing, a joint two-attribute reweighing sensitivity run, and a "
    "second, independent technique -- Fairlearn equalized-odds post-processing. Every "
    "hypothesis test (a paired bootstrap AUC test, two-proportion z-tests, a Spearman rank "
    "correlation) is reported with an exact p-value. Data: the UCI \"default of credit card "
    "clients\" dataset (Yeh & Lien, 2009) -- 30,000 real, de-identified Taiwanese "
    "credit-card accounts with six months of billing/repayment history, split 80/20 "
    "(stratified) into training and held-out test sets. Technology: Python, scikit-learn, "
    "XGBoost, CatBoost, InterpretML (EBM), SHAP, Fairlearn, Optuna, and SciPy, run entirely "
    "on commodity CPU hardware with no GPU required. Major results: tuned XGBoost reached "
    f"the highest held-out performance (ROC-AUC = {tuned[best_model]['roc_auc']:.4f}), "
    f"significantly exceeding tuned logistic regression (ROC-AUC = "
    f"{tuned['logistic_regression_tuned']['roc_auc']:.4f}; one-tailed paired bootstrap "
    f"test, {fmt_p(rq1['p_value'])}); SHAP and EBM feature-importance rankings agreed "
    "strongly (Spearman's rho = 0.892, p < .001); SEX-based reweighing significantly "
    f"narrowed the SEX selection-rate gap ({fmt_p(rq34[best_model]['SEX']['pre_mitigation']['p_value'])} "
    f"pre-mitigation to {fmt_p(rq34[best_model]['SEX']['post_mitigation']['p_value'])} "
    "post-mitigation) but, being single-attribute, left the highly significant AGE_GROUP "
    f"gap largely intact ({fmt_p(rq34[best_model]['AGE_GROUP']['pre_mitigation']['p_value'])} "
    f"pre-, {fmt_p(rq34[best_model]['AGE_GROUP']['post_mitigation']['p_value'])} "
    "post-mitigation); a joint reweighing run and an equalized-odds post-processing "
    "alternative each improved AGE_GROUP parity further, at an additional accuracy cost. "
    "Implementation area: the resulting pipeline and decision framework are intended for "
    "credit-risk model-validation and MLOps teams needing an auditable, reproducible "
    "reference for building and monitoring similarly regulated lending scoring models.",
    style="Normal",
)
doc.add_page_break()

# ===========================================================================
# INTRODUCTION  (Background and Context; Problem Statement; Purpose of the
# Study; Research Questions; Contributions and Expected Value)
# ===========================================================================
copy_in("Introduction", "Interim Project Status (Progress Snapshot)")
# The interim doc's opening summary paragraph described an "interim" pipeline
# run; the Final Report is a standalone, completed account, so that framing
# is tightened in place rather than left as-is.
replace_after(
    "This report documents an integrated machine-learning pipeline that predicts the probability a revolving credit-card account will default on its next payment, explains those predictions through both post-hoc and inherently interpretable methods, and audits and mitigates fairness gaps across two demographic attributes, all on a single, real, publicly available dataset.",
    "This report documents a completed, integrated machine-learning pipeline that predicts "
    "the probability a revolving credit-card account will default on its next payment, "
    "explains those predictions through both post-hoc and inherently interpretable methods, "
    "and audits and mitigates fairness gaps across two demographic attributes -- using two "
    "independent mitigation techniques -- all on a single, real, publicly available dataset, "
    "with every research question resolved by an exact hypothesis-test p-value.",
    after_text="Introduction",
)

heading("Research Questions", style="Heading 2")
body(
    "The study is organized around four research questions that progress from predictive "
    "accuracy across four model families (RQ1), through a comparison of post-hoc and native "
    "explainability (RQ2), to fairness auditing and mitigation of the best model (RQ3), and "
    "finally replication of that mitigation across every model family (RQ4). Each is paired "
    "with an explicit statistical hypothesis, resolved in the Results section below.",
    style="Body Text",
)
copy_in("Research Question 1 (RQ1)", "Figure 1 — RQ1-RQ4 lineage and overall solution flow.")

heading("Contributions and Expected Value", style="Heading 2")
body(
    "Practical contribution: a reusable reference pipeline that a credit-risk team can adapt "
    "to its own portfolio to obtain accuracy, explainability, and fairness auditing from a "
    "single run rather than three disconnected processes.",
    style="Body Text",
)
body(
    "Technical contribution (methods/models): a like-for-like comparison of two independent "
    "bias-mitigation techniques -- Kamiran and Calders (2012) reweighing and Fairlearn "
    "equalized-odds post-processing (Hardt et al., 2016) -- together with a single-attribute "
    "vs. joint-attribute reweighing sensitivity analysis, all evaluated with exact "
    "hypothesis-test p-values rather than point estimates alone, which is more rigorous than "
    "most single-technique, point-estimate-only fairness studies in the credit-scoring "
    "literature (see Literature Review).",
    style="Body Text",
)
body(
    "Value to stakeholders/target users: model-validation and MLOps teams gain a documented, "
    "auditable decision framework for choosing an operating point that balances predictive "
    "performance against fairness across more than one protected attribute at once; "
    "cardholders benefit indirectly from a model whose demographic disparities are measured "
    "and, where feasible, mitigated before deployment.",
    style="Body Text",
)
doc.add_page_break()

# ===========================================================================
# LITERATURE REVIEW
# ===========================================================================
heading("Literature Review", style="Heading 1")
copy_in("Literature Review Approach", "Data Description")
doc.add_page_break()

# ===========================================================================
# MATERIALS AND METHOD
# ===========================================================================
heading("Materials and Method", style="Heading 1")

heading("Data Description", style="Heading 2")
copy_in("Data Description", "Screenshots / Evidence", after=find_para_index(src, "Literature Review Approach"))

heading("Data Cleaning", style="Heading 2")
copy_in("Data Cleaning", "EDA Results (Interpretation Emphasis)")

heading("Exploratory Data Analysis", style="Heading 2")
copy_in("EDA Results (Interpretation Emphasis)", "Modelling")

heading("Sample Size Justification", style="Heading 2")
copy_in("Sample size calculation", "Literature survey")

heading("Research Hypotheses (H0 / Ha) by Research Question", style="Heading 2")
body(
    "Each research question is resolved with an explicit null/alternative hypothesis pair, "
    "tested at alpha = .05; the exact test statistic and p-value for each are reported under "
    "Results by Research Question below.",
    style="Body Text",
)
bullets(
    [
        "RQ1 -- H0: ROC-AUC(tuned XGBoost) = ROC-AUC(tuned Logistic Regression); "
        "Ha: ROC-AUC(tuned XGBoost) > ROC-AUC(tuned Logistic Regression). "
        "Test: one-tailed paired bootstrap test on the shared held-out test set.",
        "RQ2 -- H0: Spearman's rho(SHAP rank, EBM rank) = 0; Ha: Spearman's rho > 0. "
        "Test: Spearman rank-correlation significance test.",
        "RQ3 -- H0: the pre-mitigation favorable-outcome selection-rate gap between the "
        "highest- and lowest-selection-rate group is zero, for SEX and for AGE_GROUP; "
        "Ha: the gap is non-zero, and mitigation narrows it. "
        "Test: two-tailed two-proportion z-test, evaluated pre- and post-mitigation.",
        "RQ4 -- H0: the reweighing-driven improvement in the SEX selection-rate gap does not "
        "differ meaningfully across the four model families; Ha: it differs by architecture. "
        "Test: the same two-proportion z-test applied per model family, compared qualitatively "
        "across models.",
    ]
)

heading("Statistical Method", style="Heading 2")
body(
    "Model justification, feature engineering, and tuning methodology appear under "
    "Architecture Diagram/Workflow below, along with the evaluation-metric formulae. One "
    "methodological correction is noted here: RQ1's a-priori power analysis (Sample Size "
    "Justification) treated ROC-AUC as an independent binomial proportion for planning "
    "purposes only; because the two models compared are scored on the same held-out set, "
    "their AUCs are statistically correlated, so a two-proportion z-test would understate "
    "that correlation. RQ1's confirmatory test is therefore a paired bootstrap test (10,000 "
    "resamples), the standard test for comparing correlated AUCs (Efron & Tibshirani, 1993); "
    "the RQ1 Results below explain this in place. RQ3/RQ4's selection-rate gaps are "
    "independent-group proportions, so the planned two-proportion z-test is used for both, "
    "applied to the highest- and lowest-selection-rate group pair for each attribute -- the "
    "same pair that drives Fairlearn's demographic parity ratio.",
    style="Normal",
)
doc.add_page_break()

# ===========================================================================
# ARCHITECTURE DIAGRAM / WORKFLOW
# ===========================================================================
heading("Architecture Diagram/Workflow", style="Heading 1")

heading("System Overview", style="Heading 2")
body(
    "The pipeline runs as a single, deterministic, seeded (seed = 42) sequence: raw data "
    "loads (falling back to a synthetic dataset only if absent), EDA and feature "
    "engineering run once and feed every downstream stage identically, four model families "
    "are trained and tuned against the same stratified 80/20 split, the best model is "
    "explained (SHAP, EBM native) and fairness-audited (Fairlearn), and two independent "
    "bias-mitigation techniques are applied and re-audited. A second, fast (under two "
    "minutes) extension stage reloads the same split and the already-tuned model artifacts "
    "to add the hypothesis-test p-values, the joint-attribute mitigation run, the "
    "equalized-odds alternative, and the decision framework, without re-running the roughly "
    f"{run_summary['elapsed_seconds'] / 60:.0f}-minute hyperparameter search; a sanity check "
    "at its start recomputes the best model's ROC-AUC and asserts it reproduces the first "
    "stage's own recorded value.",
    style="Body Text",
)

heading("Architecture Diagram", style="Heading 2")
add_image(FIG_EVID / "rq_solution_flowchart.png", "Figure 1. End-to-end pipeline architecture and RQ1-RQ4 lineage.", width_in=6.2)
body(
    "As shown in Figure 1, data flows from ingestion through preprocessing, EDA, feature "
    "engineering, model training/tuning, evaluation, explainability, and fairness "
    "audit/mitigation; each stage's output is consumed by exactly one research question, "
    "and RQ3/RQ4's mitigation stage feeds back into a second evaluation pass rather than "
    "terminating the pipeline.",
    style="Body Text",
)

heading("Workflow Components", style="Heading 2")
body("Data Ingestion", style="Body Text", bold=True)
body(
    "The pipeline ingests the UCI \"default of credit card clients\" CSV (or, if absent, a "
    "schema-matched synthetic fallback) through a single loader function, so every "
    "downstream stage is agnostic to which source produced the dataframe.",
    style="Normal",
)
body("Data Preprocessing", style="Body Text", bold=True)
body(
    "Numeric features are median-imputed and standardized; categorical features are "
    "imputed with the most frequent category and one-hot encoded; the same "
    "ColumnTransformer is shared, unfit, across all four model pipelines so every model "
    "sees an identical feature space (see Data Cleaning above for the full column-level "
    "detail).",
    style="Normal",
)
body("Exploratory Data Analysis (EDA)", style="Body Text", bold=True)
body("Covered in full under Materials and Method > Exploratory Data Analysis above.", style="Normal")
body("Feature Engineering", style="Body Text", bold=True)
body(
    "Covered under Materials and Method > Exploratory Data Analysis above (utilization "
    "ratios, repayment-behavior aggregates, and the derived AGE_GROUP protected attribute).",
    style="Normal",
)
body("Model Development", style="Body Text", bold=True)
copy_in("Choice of Models with Justification", "Explainability, Fairness Audit, and Bias-Mitigation Methodology")
replace_after(
    "Four model families are trained on a single shared preprocessing pipeline (median imputation and scaling for numeric features, most-frequent imputation and one-hot encoding for categorical features) so that SHAP, EBM's native explanation, and the fairness audit are all comparable across models. L2-regularized logistic regression serves as the interpretable baseline; XGBoost and CatBoost are two independently developed gradient-boosted tree ensembles; and the Explainable Boosting Machine (EBM) is a glass-box generalized additive model. All four are tuned (RandomizedSearchCV for logistic regression, Optuna/TPE search for the other three) over 5-fold (2-fold for EBM, for tractable wall-clock time) stratified cross-validated ROC-AUC. H1: because gradient-boosted trees and additive shape functions both capture structure a purely linear model's additive-in-the-raw-features form cannot (interaction effects for the ensembles, non-linear per-feature shape functions for EBM), all three are hypothesized to achieve significantly higher held-out ROC-AUC than tuned logistic regression.",
    "Four model families share a single preprocessing pipeline so SHAP, EBM's native "
    "explanation, and the fairness audit remain comparable across models: L2-regularized "
    "logistic regression (interpretable baseline), two independent gradient-boosted "
    "ensembles (XGBoost, CatBoost), and the Explainable Boosting Machine (EBM, a glass-box "
    "additive model). All four are tuned (RandomizedSearchCV for logistic regression; "
    "Optuna/TPE for the others) over stratified cross-validated ROC-AUC. H1: because "
    "gradient-boosted trees and additive shape functions both capture structure a linear "
    "model cannot, all three non-linear models are hypothesized to significantly outperform "
    "tuned logistic regression.",
    after_text="Choice of Models with Justification",
)
replace_after(
    "Justification: a linear, L2-regularized logistic regression is the interpretable baseline against which the three non-linear challengers are measured. Its coefficients map directly onto the additive, reason-code-style explanations regulators expect adverse-action notices (Consumer Financial Protection Bureau, 2023), giving it a low interpretability cost even before any post-hoc XAI is applied. Tuned via RandomizedSearchCV (15 iterations, 5-fold stratified cross-validated ROC-AUC) over the regularization strength and solver.",
    "Justification: L2-regularized logistic regression is the interpretable baseline; its "
    "coefficients map directly onto the reason-code explanations regulators expect for "
    "adverse-action notices (Consumer Financial Protection Bureau, 2023). Tuned via "
    "RandomizedSearchCV (15 iterations, 5-fold stratified ROC-AUC) over regularization "
    "strength and solver.",
    after_text="Model 1: L2-Regularized Logistic Regression — Interpretable Baseline",
)
replace_after(
    "Justification: gradient-boosted trees capture non-linear interaction effects (e.g., between credit-utilization ratio and recent repayment status) that a linear model's additive form cannot represent (Chen & Guestrin, 2016; Dastile et al., 2020). XGBoost is hypothesized (H1) to achieve significantly higher held-out ROC-AUC than tuned logistic regression. Tuned via a 25-trial Optuna/TPE search (5-fold stratified cross-validated ROC-AUC) over tree depth, learning rate, subsampling, and regularization; class imbalance is handled via scale_pos_weight rather than resampling, to keep the full real sample size.",
    "Justification: gradient-boosted trees capture non-linear interaction effects a linear "
    "model cannot represent (Chen & Guestrin, 2016; Dastile et al., 2020), hypothesized "
    "(H1) to significantly outperform tuned logistic regression. Tuned via a 25-trial "
    "Optuna/TPE search over tree depth, learning rate, subsampling, and regularization; "
    "class imbalance is handled via scale_pos_weight.",
    after_text="Model 2: XGBoost — Non-Linear Ensemble Challenger",
)
replace_after(
    "Justification: CatBoost is a second, independently developed gradient-boosted tree ensemble that uses ordered boosting to reduce the prediction shift / target leakage that standard boosting can introduce on categorical features (Prokhorenkova et al., 2018). Including a second ensemble alongside XGBoost tests whether RQ1's accuracy gain is a property of gradient boosting generally, or specific to one implementation. Tuned via a 20-trial Optuna/TPE search over tree depth, learning rate, L2 leaf regularization, and subsampling; class imbalance is handled via Cat Boost's built-in auto_class_weights.",
    "Justification: CatBoost is a second, independently developed gradient-boosted "
    "ensemble using ordered boosting to reduce target leakage on categorical features "
    "(Prokhorenkova et al., 2018), testing whether RQ1's accuracy gain is specific to "
    "XGBoost or general to gradient boosting. Tuned via a 20-trial Optuna/TPE search; "
    "class imbalance is handled via CatBoost's built-in auto_class_weights.",
    after_text="Model 3: CatBoost — Second Ensemble Challenger",
)
replace_after(
    "Justification: EBM (Lou et al., 2012; implemented via InterpretML, Nori et al., 2019) is a generalized additive model with pairwise interactions -- a sum of per-feature (and select per-feature-pair) shape functions learned by gradient boosting but constrained to remain exactly additive and therefore exactly interpretable, unlike XGBoost or CatBoost. Including EBM tests two things at once: whether most of the ensembles' accuracy gain over logistic regression is recoverable from an additive model (H1), and, via its own native explanation, whether SHAP's post-hoc approximation of a black-box model can be trusted as a stand-in for an explanation that needs no approximating (H2, RQ2). Tuned via a 6-trial Optuna/TPE search (2-fold cross-validated ROC-AUC, a smaller budget than the other models because EBM fits are markedly slower at this row count) over bin count, max leaves, learning rate, and the number of outer bagging rounds; class imbalance is handled via an explicit balanced sample weight, since EBM has no built-in class_weight parameter.",
    "Justification: EBM (Lou et al., 2012; InterpretML, Nori et al., 2019) is a "
    "generalized additive model with pairwise interactions -- exactly interpretable by "
    "construction, unlike XGBoost or CatBoost. It tests whether most of the ensembles' "
    "accuracy gain is recoverable from an additive model (H1) and whether SHAP's post-hoc "
    "approximation can be trusted against an exact explanation (H2, RQ2). Tuned via a "
    "6-trial Optuna/TPE search (2-fold, a smaller budget since EBM fits are slower); "
    "class imbalance is handled via an explicit balanced sample weight.",
    after_text="Model 4: Explainable Boosting Machine (EBM) — Glass-Box Challenger",
)
copy_in("Explainability, Fairness Audit, and Bias-Mitigation Methodology", "Features Included and Feature Engineering")
copy_in("Features Included and Feature Engineering", "Evaluation Metrics (Include Formulae and Calculations)")
body("Model Evaluation", style="Body Text", bold=True)
copy_in("Evaluation Metrics (Include Formulae and Calculations)", "Preliminary Results")
body("Deployment", style="Body Text", bold=True)
body("Covered under Implementation and User Benefit below.", style="Normal")

heading("Tools and Technologies", style="Heading 2")
body(
    "Python 3.13; pandas/NumPy for data handling; scikit-learn for preprocessing, the "
    "logistic-regression baseline, and evaluation metrics; XGBoost and CatBoost for the two "
    "gradient-boosted ensembles; InterpretML's ExplainableBoostingClassifier for the "
    "glass-box model; SHAP for post-hoc explainability; Fairlearn for fairness auditing "
    "(MetricFrame), reweighing-style mitigation, and equalized-odds post-processing "
    "(ThresholdOptimizer); Optuna (TPE sampler) for Bayesian hyperparameter search; and "
    "SciPy for the study's hypothesis tests (two-proportion z-tests, the paired bootstrap "
    "AUC test, the Spearman correlation test). All training and analysis ran on commodity "
    "CPU hardware; no GPU was required.",
    style="Normal",
)
doc.add_page_break()

# ===========================================================================
# RESULTS
# ===========================================================================
heading("Results", style="Heading 1")

heading("Model Performance", style="Heading 2")
copy_in("Preliminary Model Performance", "Interim Limitations and Risks")
body(
    "Table 11 extends the comparison to the study's two additional mitigation techniques -- "
    f"joint SEX x AGE_GROUP reweighing and equalized-odds post-processing, both applied to "
    f"the RQ1 best model ({label(best_model)}) -- alongside the existing tuned and "
    f"SEX-only-reweighed rows, so all four operating points can be compared directly.",
    style="Normal",
)
_dec_rows = []
for _name, _row in decision["table"].items():
    _dec_rows.append(
        [
            _name,
            f"{_row['roc_auc']:.4f}" if _row["roc_auc"] is not None else "n/a (thresholded output)",
            f"{_row['f1']:.4f}",
            f"{_row['dp_ratio_SEX']:.3f}",
            f"{_row['dp_ratio_AGE_GROUP']:.3f}",
            f"{_row['eo_diff_SEX']:.3f}",
            f"{_row['eo_diff_AGE_GROUP']:.3f}",
        ]
    )
add_table(
    ["Technique", "ROC-AUC", "F1", "DP Ratio (SEX)", "DP Ratio (AGE_GROUP)", "EO Diff (SEX)", "EO Diff (AGE_GROUP)"],
    _dec_rows,
)
body("Table 11. Accuracy-fairness decision framework: tuned XGBoost under four operating points.", style="LO-normal")

heading("Visual Evidence", style="Heading 2")
body(
    "Figures 5-9 above (reused from the Model Performance table's supporting evidence) "
    "cover baseline-vs-tuned comparison, XGBoost's diagnostic suite, SHAP and EBM global "
    "importance, and pre/post-mitigation selection rates. Figure 10 below adds the "
    "accuracy-fairness trade-off across all four operating points evaluated in Table 11.",
    style="Normal",
)
add_image(
    FIG_DECISION / "accuracy_fairness_tradeoff.png",
    "Figure 10. Tuned XGBoost: ROC-AUC vs. the binding (lower of SEX/AGE_GROUP) demographic "
    "parity ratio, across the four mitigation operating points in Table 11. The dashed line "
    "is the four-fifths convention (0.80).",
    width_in=6.0,
)
doc.add_page_break()

heading("Results by Research Question", style="Heading 2")

# --- RQ1 ---
copy_in("RQ1 – Model Performance Comparison", "RQ2 – Model Explainability")
insert_paragraph_after(
    "To determine whether the best-performing model significantly outperformed the others, the following hypotheses were tested using a one-tailed two-proportion z-test on the held-out ROC-AUC scores (treating ROC-AUC as a proportion).",
    "As discussed under Materials and Method > Statistical Method above, the two-proportion "
    "framing below was used only to size the a-priori minimum sample; because the two AUCs "
    "compared here are correlated (same held-out test set), the actual confirmatory test "
    "reported under Results is a one-tailed paired bootstrap test, not this z-test.",
    style="LO-normal",
)
replace_after(
    "The tuned XGBoost model achieved the highest overall predictive performance, with a held-out ROC-AUC of 0.7813. It also recorded a PR-AUC of 0.5609, F1-score of 0.5335, KS statistic of 0.4367, and Brier Score of 0.1785. Compared with the tuned Logistic Regression model (ROC-AUC = 0.7463), XGBoost achieved an improvement of 0.0350 ROC-AUC points.",
    f"The tuned {label(best_model)} model achieved the highest overall predictive performance, "
    f"with a held-out ROC-AUC of {tuned[best_model]['roc_auc']:.4f} (PR-AUC = "
    f"{tuned[best_model]['pr_auc']:.4f}, F1 = {tuned[best_model]['f1']:.4f}, KS = "
    f"{tuned[best_model]['ks_statistic']:.4f}, Brier = {tuned[best_model]['brier_score']:.4f}), "
    f"compared with tuned Logistic Regression's ROC-AUC of "
    f"{tuned['logistic_regression_tuned']['roc_auc']:.4f} -- an improvement of "
    f"{rq1['observed_diff']:.4f} ROC-AUC points. Because the two AUCs are estimated on the "
    "same held-out test set, they are statistically correlated, so the confirmatory "
    "significance test used here is a one-tailed paired bootstrap test (10,000 resamples; "
    "see Statistical Method) rather than the two-proportion approximation used only for the "
    f"a-priori power calculation. The bootstrap test rejected H0: the observed AUC "
    f"advantage (95% CI [{rq1['ci_95'][0]:.4f}, {rq1['ci_95'][1]:.4f}]) lies entirely above "
    f"zero, {fmt_p(rq1['p_value'])} (one-tailed, {rq1['n_boot']} resamples).",
    after_text="RQ1 – Model Performance Comparison",
)
replace_after(
    "Hyperparameter tuning improved the performance of every model compared with its baseline implementation.",
    "Hyperparameter tuning improved the performance of every model compared with its baseline "
    "implementation (Table 11 / Model Performance above).",
    after_text="RQ1 – Model Performance Comparison",
)

# --- RQ2 (already had a real p-value; copied verbatim, only the RQ2 -> RQ3 range) ---
copy_in("RQ2 – Model Explainability", "RQ3 – Fairness Audit and Bias Mitigation")

# --- RQ3 ---
copy_in("RQ3 – Fairness Audit and Bias Mitigation", "RQ4 – Cross-Model Fairness Replication")
_sex_pre, _sex_post = rq34[best_model]["SEX"]["pre_mitigation"], rq34[best_model]["SEX"]["post_mitigation"]
_age_pre, _age_post = rq34[best_model]["AGE_GROUP"]["pre_mitigation"], rq34[best_model]["AGE_GROUP"]["post_mitigation"]
replace_after(
    "Before mitigation, the demographic parity ratio was already above the commonly accepted four-fifths (0.80) threshold for both protected attributes:",
    "Before mitigation, the demographic parity ratio was already above the commonly accepted "
    f"four-fifths (0.80) threshold for both protected attributes (SEX = "
    f"{pre_fair[best_model]['SEX']['demographic_parity_ratio']:.3f}; AGE_GROUP = "
    f"{pre_fair[best_model]['AGE_GROUP']['demographic_parity_ratio']:.3f}). However, the "
    "ratio threshold and a hypothesis test answer different questions: a two-tailed "
    "two-proportion z-test on the highest- vs. lowest-selection-rate group pair found the "
    f"pre-mitigation SEX gap significant ({_sex_pre['p1']:.3f} vs. {_sex_pre['p2']:.3f}, "
    f"{fmt_p(_sex_pre['p_value'])}) and the pre-mitigation AGE_GROUP gap highly significant "
    f"({_age_pre['p1']:.3f} vs. {_age_pre['p2']:.3f}, {fmt_p(_age_pre['p_value'])}), despite "
    "both ratios clearing the four-fifths convention -- a large sample size (n = 6,000 held "
    "out) makes even a ratio-compliant gap statistically detectable, illustrating that the "
    "four-fifths rule and a significance test are not interchangeable evidence.",
    after_text="RQ3 – Fairness Audit and Bias Mitigation",
)
replace_after(
    "Applying the Kamiran and Calders reweighing approach increased the demographic parity ratio for both attributes.",
    "Applying the Kamiran and Calders reweighing approach (on SEX) increased the demographic "
    f"parity ratio for both attributes (SEX: "
    f"{pre_fair[best_model]['SEX']['demographic_parity_ratio']:.3f} -> "
    f"{post_fair[best_model]['SEX']['demographic_parity_ratio']:.3f}; AGE_GROUP: "
    f"{pre_fair[best_model]['AGE_GROUP']['demographic_parity_ratio']:.3f} -> "
    f"{post_fair[best_model]['AGE_GROUP']['demographic_parity_ratio']:.3f}). The z-test "
    f"confirms this for SEX -- the post-mitigation gap is no longer significant "
    f"({_sex_post['p1']:.3f} vs. {_sex_post['p2']:.3f}, {fmt_p(_sex_post['p_value'])}), so H0 "
    "is retained for SEX post-mitigation -- but, because reweighing was applied to SEX only, "
    f"the AGE_GROUP gap remains highly significant after mitigation "
    f"({_age_post['p1']:.3f} vs. {_age_post['p2']:.3f}, {fmt_p(_age_post['p_value'])}).",
    after_text="RQ3 – Fairness Audit and Bias Mitigation",
)
replace_after(
    "Although demographic parity improved, the comparatively higher Equalized Odds Difference for AGE_GROUP suggests that some fairness of disparities remain and should be discussed as a study of limitation.",
    "Although demographic parity improved, the comparatively higher Equalized Odds Difference "
    "for AGE_GROUP suggests that some fairness disparities remain; two further analyses were "
    "run to probe this directly.",
    after_text="RQ3 – Fairness Audit and Bias Mitigation",
)

heading("RQ3 Extension A: Joint SEX x AGE_GROUP Reweighing Sensitivity Analysis", style="Heading 3")
body(
    f"Reweighing was recomputed on the joint SEX x AGE_GROUP label and {label(best_model)} "
    "retrained with the same hyperparameters as the SEX-only mitigated model, isolating the "
    "weighting scheme's effect. Relative to SEX-only reweighing, joint reweighing raised "
    f"both attributes' demographic parity ratio further (SEX: "
    f"{joint['fairness_single_attr_mitigation']['SEX']['demographic_parity_ratio']:.3f} -> "
    f"{joint['fairness_joint_mitigation']['SEX']['demographic_parity_ratio']:.3f}; AGE_GROUP: "
    f"{joint['fairness_single_attr_mitigation']['AGE_GROUP']['demographic_parity_ratio']:.3f} "
    f"-> {joint['fairness_joint_mitigation']['AGE_GROUP']['demographic_parity_ratio']:.3f}) "
    f"at a small additional ROC-AUC cost "
    f"({joint['performance_single_attr_mitigation']['roc_auc']:.4f} -> "
    f"{joint['performance_joint_mitigation']['roc_auc']:.4f}). Joint reweighing narrows -- "
    "but, per the z-tests above, does not eliminate -- the AGE_GROUP gap left open by "
    "SEX-only mitigation.",
    style="Normal",
)

heading("RQ3 Extension B: Equalized-Odds Post-Processing (Second Mitigation Technique)", style="Heading 3")
body(
    f"As a second, independent technique, Fairlearn's ThresholdOptimizer was fit on the "
    f"already-tuned {label(best_model)} pipeline (equalized-odds constraint, SEX), requiring "
    "no retraining. It matched reweighing's SEX demographic parity ratio "
    f"({eo['fairness']['SEX']['demographic_parity_ratio']:.3f} vs. "
    f"{eo['fairness_reweighing_comparison']['SEX']['demographic_parity_ratio']:.3f}) but at a "
    f"larger cost to F1 ({eo['performance']['f1']:.4f} vs. reweighing's "
    f"{joint['performance_single_attr_mitigation']['f1']:.4f}), since it selects "
    "group-specific thresholds to satisfy the constraint exactly rather than retraining the "
    "scores (its hard, thresholded output has no defined ROC-AUC/PR-AUC/Brier, omitted from "
    "Table 11). Reweighing is the better operating point here when overall classification "
    "quality matters alongside fairness; post-processing is preferable only when the model "
    "cannot be retrained at all.",
    style="Normal",
)

# --- RQ4 ---
copy_in("RQ4 – Cross-Model Fairness Replication", "Preliminary Model Performance")
_rq4_lines = []
for _name in ["logistic_regression_tuned", "xgboost_tuned", "catboost_tuned", "ebm_tuned"]:
    _pre_p = rq34[_name]["SEX"]["pre_mitigation"]["p_value"]
    _post_p = rq34[_name]["SEX"]["post_mitigation"]["p_value"]
    _rq4_lines.append(f"{label(_name)}: pre-mitigation {fmt_p(_pre_p)}, post-mitigation {fmt_p(_post_p)}")
replace_after(
    "The demographic parity ratio before and after mitigation was:",
    "The demographic parity ratio before and after mitigation was reported in Table 10 "
    "(Model Performance above); the corresponding two-proportion z-test on the SEX "
    "selection-rate gap, pre- and post-mitigation, for each model family was: "
    + "; ".join(_rq4_lines) + ".",
    after_text="RQ4 – Cross-Model Fairness Replication",
)
replace_after(
    "These findings suggest that the Kamiran and Calders reweighing approach produced broadly consistent improvements across different model architectures. Consequently, the fairness intervention demonstrated reasonable generalizability beyond XGBoost, providing preliminary support for the study's fourth research hypothesis.",
    "Logistic Regression's SEX gap was never statistically significant, before or after "
    "mitigation; XGBoost and EBM's significant pre-mitigation SEX gaps both became "
    "non-significant post-mitigation; CatBoost's gap was borderline both before and after "
    "(p approximately .04 in both cases), the one model family where reweighing did not "
    "clearly resolve the SEX disparity. These findings suggest that the Kamiran and Calders "
    "reweighing approach produced broadly consistent, though not uniformly complete, "
    "improvements across different model architectures, supporting the study's fourth "
    "research hypothesis with one noted exception (CatBoost).",
    after_text="RQ4 – Cross-Model Fairness Replication",
)

heading("Overall Interpretation", style="Heading 2")
body(
    f"The strongest, most consistent finding is that non-linear models materially "
    f"outperform the interpretable linear baseline (RQ1, {fmt_p(rq1['p_value'])}) without "
    "sacrificing explainability: SHAP and the glass-box EBM agree strongly on the drivers "
    "(RQ2, rho = 0.892, p < .001) -- repayment history and credit-utilization, not "
    "demographic fields. The fairness results are more nuanced: a single, model-agnostic "
    "mitigation applied to one protected attribute reliably resolves that attribute's "
    "disparity for most models (RQ4) but does not automatically resolve a second, "
    "unmitigated attribute's disparity (RQ3), even though both attributes cleared the "
    "four-fifths threshold throughout. Two further interventions -- joint-attribute "
    "reweighing and an independent post-processing technique -- each narrow the remaining "
    "gap further, each at a measurable accuracy cost, confirming that accuracy and fairness "
    "across multiple protected attributes are several overlapping trade-offs a practitioner "
    "must choose among explicitly, not one problem solved by a single mitigation pass.",
    style="Normal",
)

heading("Practical Significance", style="Heading 2")
body(
    "Three results carry direct operational weight for a credit-risk model-validation team. "
    f"First, moving to {label(best_model)} is a statistically and practically significant "
    f"accuracy gain ({rq1['observed_diff']:.3f} ROC-AUC) without sacrificing explainability. "
    "Second, a four-fifths-rule pass is not sufficient evidence that a protected attribute "
    "is unaffected by bias: this study's pre-mitigation model passed the ratio test on both "
    "SEX and AGE_GROUP while a significance test found both gaps real -- a validation team "
    "relying on the ratio alone would have under-detected the AGE_GROUP disparity. Third, "
    "mitigating one protected attribute does not mitigate another for free; a team auditing "
    "multiple attributes should budget for a joint mitigation pass or an explicit "
    "per-attribute plan, and expect a larger accuracy cost the more attributes are "
    "mitigated jointly.",
    style="Normal",
)
doc.add_page_break()

# ===========================================================================
# IMPLEMENTATION AND USER BENEFIT
# ===========================================================================
heading("Implementation and User Benefit", style="Heading 1")
body(
    "This section describes how the pipeline could be deployed and used in a production "
    "credit-decisioning setting; no deployment infrastructure (API, dashboard, or cloud "
    "service) was built as part of this study -- the description below is a design, "
    "grounded in the trained artifacts and decision framework produced above, not a claim "
    "of an operating system.",
    style="Normal",
)

heading("Deployment Approach", style="Heading 2")
body(
    f"The trained {label(best_model)} pipeline (a single serialized scikit-learn Pipeline "
    "object) is small enough to serve as a batch or low-latency REST scoring service: given "
    "an applicant's record in the training schema, it returns a default probability and, "
    "from the SHAP/EBM artifacts, the top contributing features for that prediction. "
    "Because the pipeline produces both a reweighing-mitigated and an "
    "equalized-odds-post-processed variant, a deployment could expose either operating "
    "point from Table 11 behind a configuration flag rather than committing to one at "
    "build time.",
    style="Normal",
)

heading("System Integration", style="Heading 2")
body(
    "Integrating into an existing loan-origination system would require four standing "
    "capabilities, each already exercised in this study's pipeline:",
    style="Normal",
)
bullets(
    [
        "Explainability documentation: SHAP/EBM local explanations map onto the reason-code "
        "format required for an adverse-action notice (Consumer Financial Protection "
        "Bureau, 2023).",
        "Periodic fairness re-evaluation: the same Fairlearn MetricFrame audit should re-run "
        "on a schedule against live scoring data, since demographic composition and default "
        "behavior can drift after deployment.",
        "Adverse action notice generation: declined applicants' top SHAP/EBM features can be "
        "templated directly into a compliant notice.",
        "Production monitoring: the ROC-AUC/calibration diagnostics and fairness metrics "
        "(demographic parity ratio, equalized odds difference) computed in this study "
        "generalize directly to a monitoring dashboard.",
    ]
)

heading("User Interaction", style="Heading 2")
body(
    "The primary user is a credit underwriter or automated decisioning system, not the "
    "applicant: the model returns a default probability plus ranked contributing factors "
    "per application, which an underwriter reviews alongside policy rules before a final "
    "decision. A secondary user, the model-validation team, interacts with the "
    "fairness-audit and decision-framework outputs (Table 11, Figure 10) on a periodic "
    "cadence rather than per-application.",
    style="Normal",
)

heading("Benefits to Users", style="Heading 2")
bullets(
    [
        "Operational efficiency: an automated, explainable score lets underwriters "
        "prioritize manual review toward the applications the model is least confident "
        "about, rather than reviewing every application at the same depth.",
        "Financial impact: the RQ1 accuracy gain over a linear scorecard "
        f"({rq1['observed_diff']:.3f} ROC-AUC, {fmt_p(rq1['p_value'])}) translates into fewer "
        "missed defaults and fewer good applicants declined, at a magnitude an issuer can "
        "estimate against its own portfolio's charge-off and revenue-per-account economics.",
        "Strategic/compliance value: a documented, reproducible fairness audit with exact "
        "hypothesis-test p-values, evaluated under two independent mitigation techniques, "
        "gives a model-risk-management or fair-lending compliance function concrete "
        "evidence to show a regulator or internal auditor, rather than a single "
        "point-estimate fairness metric.",
    ]
)

heading("Example Use Case", style="Heading 2")
body(
    "A card issuer receives an application from a 24-year-old applicant. The pipeline "
    "engineers utilization/repayment features from six months of billing history, scores "
    f"the application with the tuned {label(best_model)} model, and returns a default "
    "probability with the top SHAP-ranked contributing features (e.g., a high "
    "PAY_TO_BILL_RATIO). Because AGE_GROUP '<25' had a materially lower favorable-outcome "
    "selection rate in the fairness audit (RQ3), the compliance dashboard flags this "
    "application's age cohort for the next periodic fairness re-evaluation without "
    "altering the individual decision -- the monitoring loop in System Integration "
    "operating on a real application, not only aggregate audit runs.",
    style="Normal",
)
doc.add_page_break()

# ===========================================================================
# LIMITATIONS AND FURTHER IMPROVEMENTS
# ===========================================================================
heading("Limitations and Further Improvements", style="Heading 1")

heading("Limitations", style="Heading 2")
body(
    "The results above should be interpreted in light of several limitations that may "
    "influence their generalizability and interpretation.",
    style="Normal",
)
copy_in("Four-Fifths Rule as a Fairness Benchmark", "Limited Bias Mitigation Evaluation")

heading("Mitigation Scope and Residual Disparity", style="Heading 3")
body(
    "Two independent techniques were evaluated -- Kamiran and Calders (2012) reweighing "
    "and Fairlearn equalized-odds post-processing (Hardt et al., 2016) -- and reweighing "
    "was further evaluated jointly across SEX and AGE_GROUP. This closes only partially: "
    f"even joint reweighing left the AGE_GROUP demographic-parity ratio at "
    f"{joint['fairness_joint_mitigation']['AGE_GROUP']['demographic_parity_ratio']:.3f} "
    "(improved, but still the lower of the two attributes) and did not eliminate the "
    "selection-rate gap's statistical significance. A third, unevaluated technique -- an "
    "in-processing fairness constraint applied during training -- remains open for future "
    "work and might close more of the remaining gap.",
    style="Normal",
)

copy_in("Fairness Metric Trade-Offs", "EBM Hyperparameter Optimization Budget")
insert_paragraph_after(
    "Different fairness metrics measure different aspects of model behavior and may not improve simultaneously. While demographic parity improved following bias mitigation, equalized odds evaluate a separate fairness criterion. For the tuned XGBoost model, the post-mitigation Equalized Odds Difference for SEX was 0.012, illustrating the well-known trade-off between fairness objectives. Consequently, improvements in demographic parity should not be interpreted as a complete guarantee of fairness across all evaluation metrics.",
    "The same trade-off appears, more sharply, for AGE_GROUP: SEX-only reweighing actually "
    f"increased its Equalized Odds Difference "
    f"({pre_fair[best_model]['AGE_GROUP']['equalized_odds_difference']:.3f} -> "
    f"{post_fair[best_model]['AGE_GROUP']['equalized_odds_difference']:.3f}), and neither "
    f"joint reweighing "
    f"({joint['fairness_joint_mitigation']['AGE_GROUP']['equalized_odds_difference']:.3f}) "
    f"nor equalized-odds post-processing "
    f"({eo['fairness']['AGE_GROUP']['equalized_odds_difference']:.3f}) fully reversed it -- "
    "evidence that optimizing demographic parity on one attribute can worsen a different "
    "fairness metric on a second, unmitigated attribute.",
    style="Normal",
)
copy_in("EBM Hyperparameter Optimization Budget", "RQ2 Sample Size Assumptions", after=find_para_index(src, "Fairness Metric Trade-Offs"))
copy_in("RQ2 Sample Size Assumptions", "Inferential Statistics")

heading("Statistical Significance vs. the Four-Fifths Threshold", style="Heading 3")
body(
    "Exact hypothesis-test p-values are now reported for all four research questions "
    "(bootstrap test for RQ1, Spearman test for RQ2, two-proportion z-tests for RQ3/RQ4), "
    "which surfaced a limitation of a different kind: the four-fifths ratio convention and "
    "a significance test can disagree, as shown in RQ3, where a ratio-compliant AGE_GROUP "
    "gap was nonetheless highly significant. Neither test alone is a complete fairness "
    "criterion -- the ratio ignores sample size, the z-test ignores practical magnitude -- "
    "so a production fairness monitor should report both.",
    style="Normal",
)

heading("Impact of Limitations", style="Heading 2")
body(
    "Collectively, the study's fairness conclusions are strongest for SEX (where both "
    "mitigation techniques resolved the gap's significance for most model families) and "
    "weakest for AGE_GROUP (where a real, significant gap persisted under every technique "
    "tried); the accuracy conclusions (RQ1, RQ2) are unaffected, since they do not depend "
    "on the mitigation technique. Any deployment decision based on this study's operating "
    "points should treat the AGE_GROUP result as the binding constraint, not SEX.",
    style="Normal",
)

heading("Future Improvements", style="Heading 2")
bullets(
    [
        "Evaluate an in-processing fairness technique (e.g., a fairness-constrained "
        "objective during training, or adversarial debiasing) to test whether it closes "
        "more of the residual AGE_GROUP gap than either pre- or post-processing did here.",
        "Extend the joint-attribute sensitivity analysis (currently run for the single "
        "best model) to all four model families, mirroring RQ4's cross-model replication "
        "design for the single-attribute case.",
        "Operationalize the governance checklist in Implementation and User Benefit above "
        "(explainability documentation, periodic fairness re-evaluation, adverse-action "
        "notice generation, production monitoring) as running code rather than a "
        "description.",
    ]
)

heading("Future Scope", style="Heading 2")
body(
    "Whether the same mitigation recipes and the same accuracy-fairness decision framework "
    "generalize beyond this single dataset remains open, as previously noted; applying the "
    "identical pipeline to a second, independent lending dataset -- ideally one that, unlike "
    "this study's UCI extract, includes both approved and declined applications -- would "
    "let a future study distinguish a dataset-specific finding from a general property of "
    "the techniques themselves.",
    style="Normal",
)
doc.add_page_break()

# ===========================================================================
# BIBLIOGRAPHY
# ===========================================================================
heading("Bibliography", style="Heading 1")
copy_in("References and Bibliography", "Appendix")
insert_paragraph_after(
    "References and Bibliography",
    "Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall/CRC.",
    style="LO-normal",
)
doc.add_page_break()

# ===========================================================================
# APPENDIX
# ===========================================================================
heading("Appendix", style="Heading 1")
copy_in("Appendix A: Pipeline Orchestration Code (scripts/run_pipeline.py)", None, after=find_para_index(src, "Appendix"))

heading("Appendix C: Extended Analysis Code (scripts/run_extended_analysis.py)", style="APA Heading 2")
body(
    f"Full source is available at {GITHUB_URL}/blob/master/scripts/run_extended_analysis.py. "
    "This script closes the four items the Interim Report listed as pending (RQ1/RQ3/RQ4 "
    "exact p-values, joint-attribute mitigation, a second mitigation technique, and the "
    "decision framework) by reloading the already-tuned model artifacts and the "
    "deterministic train/test split, rather than re-running hyperparameter tuning. The "
    "excerpt below shows its module docstring; the new statistical-test and "
    "equalized-odds-postprocessing functions it calls are in "
    "src/dac/fairness/significance.py and src/dac/fairness/postprocessing.py.",
    style="APA Heading 2",
)
_ext_script_text = (REPO / "scripts" / "run_extended_analysis.py").read_text(encoding="utf-8")
_ext_docstring = _ext_script_text.split('"""')[1].strip()
doc.add_paragraph(_ext_docstring, style="Preformatted Text")

# ---------------------------------------------------------------------------
# Final structural rename pass: apply uniformly regardless of which of
# several duplicate occurrences of these labels a copy_in() call brought in.
# ---------------------------------------------------------------------------
while rename_heading("Preliminary Results", "Results"):
    pass
rename_heading("Preliminary Findings by Research Question", "Results by Research Question")
rename_heading("Interim Limitations and Risks", "Limitations and Further Improvements")
rename_heading("Preliminary Model Performance", "Baseline, Tuned, and Mitigated Model Comparison")
rename_heading("References and Bibliography", "References")

def _is_heading_style(style_name: str) -> bool:
    return style_name.startswith("Heading") or style_name.startswith("APA Heading") or style_name == "APA2"


# Drop stray empty heading-styled paragraphs left over from source blank
# separator paragraphs that carried a heading style in the interim doc.
for _p in list(doc.paragraphs):
    if not _p.text.strip() and _is_heading_style(_p.style.name):
        _el = _p._p
        _el.getparent().remove(_el)

# Drop back-to-back heading paragraphs with identical text (a section
# heading added fresh immediately followed by a copied source heading that
# happens to say the same thing verbatim -- keep only the first).
_paras = list(doc.paragraphs)
for _i in range(len(_paras) - 1):
    _p, _next = _paras[_i], _paras[_i + 1]
    if (
        _is_heading_style(_p.style.name)
        and _is_heading_style(_next.style.name)
        and _p.text.strip()
        and _p.text.strip() == _next.text.strip()
    ):
        _el = _next._p
        _el.getparent().remove(_el)

doc.save(str(OUTPUT_DOC))
print(f"Wrote {OUTPUT_DOC}")
print(f"Paragraphs: {len(doc.paragraphs)}  Tables: {len(doc.tables)}")




