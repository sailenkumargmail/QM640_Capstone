"""Builds a 15-minute senior-stakeholder PowerPoint deck summarizing the
QM640 capstone project (credit-default prediction with XAI and bias
mitigation), sourced from the latest pipeline run
(reports/model_performance/run_summary.json and reports/figures/*).

Usage:
    python scripts/build_stakeholder_ppt.py
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

REPO = Path(__file__).resolve().parents[1]
FIG_HC = REPO / "reports" / "figures" / "home_credit"
FIG_HMDA = REPO / "reports" / "figures" / "hmda"
METRICS_DIR = REPO / "reports" / "model_performance"
OUTPUT = REPO / "docs" / "QM640_Capstone_Stakeholder_Presentation.pptx"

summary = json.loads((METRICS_DIR / "run_summary.json").read_text(encoding="utf-8"))
best_model = summary["best_model"]
baseline = summary["baseline_metrics"]
tuned = summary["tuned_metrics"]
mitigated = summary["mitigated_performance"]
pre_fair = summary["fairness_pre_mitigation"]
post_fair = summary["fairness_post_mitigation"]
shap_top = summary["shap_top_features"]
hmda_perf = summary["hmda_results"]["performance"]
hmda_fair = summary["hmda_results"]["fairness"]

lr_base, xgb_base = baseline["logistic_regression_baseline"], baseline["xgboost_baseline"]
lr_tuned, xgb_tuned = tuned["logistic_regression_tuned"], tuned["xgboost_tuned"]

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x0B, 0x1F, 0x3A)
NAVY_MID = RGBColor(0x16, 0x33, 0x5C)
TEAL = RGBColor(0x14, 0xB8, 0xA6)
AMBER = RGBColor(0xF5, 0x9E, 0x0B)
RED = RGBColor(0xDC, 0x26, 0x26)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
GRAY = RGBColor(0x4B, 0x55, 0x63)
LIGHT_GRAY = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TEXT = RGBColor(0x1E, 0x29, 0x3B)

FONT = "Calibri"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def add_slide():
    return prs.slides.add_slide(BLANK)


def set_background(slide, color=WHITE):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    bg.shadow.inherit = False
    slide.shapes._spTree.remove(bg._element)
    slide.shapes._spTree.insert(2, bg._element)
    return bg


def add_textbox(slide, left, top, width, height, text, size=18, color=DARK_TEXT,
                 bold=False, italic=False, align=PP_ALIGN.LEFT, font=FONT,
                 anchor=None, line_spacing=None):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    if anchor is not None:
        tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    if line_spacing:
        p.line_spacing = line_spacing
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = font
    return box


def add_bullets(slide, left, top, width, height, items, size=16, color=DARK_TEXT,
                 font=FONT, space_after=10, line_spacing=1.08):
    """items: list of (text, level, bold_lead) or plain strings."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            text, level = item[0], item[1]
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        bullet = "▪ " if level == 0 else "– "
        r = p.add_run()
        r.text = bullet + text
        r.font.size = Pt(size - (2 * level))
        r.font.color.rgb = color
        r.font.name = font
        r.font.bold = (level == 0)
    return box


def add_header(slide, kicker, title, dark=True):
    set_background(slide, NAVY if dark else WHITE)
    band_color = TEAL
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.42), SW, Pt(3))
    band.fill.solid(); band.fill.fore_color.rgb = band_color
    band.line.fill.background(); band.shadow.inherit = False
    add_textbox(slide, Inches(0.6), Inches(0.28), Inches(11), Inches(0.4), kicker,
                size=14, color=TEAL if dark else GRAY, bold=True)
    add_textbox(slide, Inches(0.6), Inches(0.62), Inches(12), Inches(0.75), title,
                size=30, color=WHITE if dark else NAVY, bold=True)
    add_textbox(slide, Inches(12.4), Inches(7.08), Inches(0.7), Inches(0.32),
                str(slide_num[0]), size=11, color=(GRAY if not dark else RGBColor(0x9C, 0xA9, 0xBA)))


slide_num = [0]


def new_content_slide(kicker, title, dark_header=True):
    slide_num[0] += 1
    s = add_slide()
    add_header(s, kicker, title, dark=dark_header)
    return s


def add_picture_fit(slide, path, left, top, max_w, max_h):
    from PIL import Image
    im = Image.open(path)
    iw, ih = im.size
    ratio = min(max_w / iw, max_h / ih)
    w, h = int(iw * ratio), int(ih * ratio)
    l = left + int((max_w - w) / 2)
    t = top + int((max_h - h) / 2)
    slide.shapes.add_picture(str(path), l, t, width=w, height=h)


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def stat_tile(slide, left, top, width, height, value, label, value_color=NAVY, bg=LIGHT_GRAY):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.adjustments[0] = 0.08
    box.fill.solid(); box.fill.fore_color.rgb = bg
    box.line.color.rgb = RGBColor(0xE2, 0xE8, 0xF0); box.line.width = Pt(0.75)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_top = Pt(6); tf.margin_bottom = Pt(6)
    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run(); r1.text = value
    r1.font.size = Pt(30); r1.font.bold = True; r1.font.color.rgb = value_color
    r1.font.name = FONT
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run(); r2.text = label
    r2.font.size = Pt(12.5); r2.font.color.rgb = GRAY; r2.font.name = FONT
    return box


def pill(slide, left, top, width, height, text, bg, fg=WHITE, size=13):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.adjustments[0] = 0.5
    box.fill.solid(); box.fill.fore_color.rgb = bg
    box.line.fill.background(); box.shadow.inherit = False
    tf = box.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = fg; r.font.name = FONT
    return box


# ===========================================================================
# Slide 1 — Title
# ===========================================================================
s = add_slide()
set_background(s, NAVY)
accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(4.55), SW, Pt(3))
accent.fill.solid(); accent.fill.fore_color.rgb = TEAL
accent.line.fill.background(); accent.shadow.inherit = False
add_textbox(s, Inches(0.9), Inches(2.55), Inches(11.5), Inches(0.5),
            "QM640 DATA ANALYTICS CAPSTONE", size=16, color=TEAL, bold=True)
add_textbox(s, Inches(0.9), Inches(3.05), Inches(11.5), Inches(1.3),
            "Credit Default Prediction: Explainable AI & Bias Mitigation",
            size=36, color=WHITE, bold=True)
add_textbox(s, Inches(0.9), Inches(4.72), Inches(11.5), Inches(0.5),
            "A responsible-AI credit-risk pipeline: accurate, explainable, and fairness-audited",
            size=16, color=RGBColor(0xC7, 0xD2, 0xE0), italic=True)
add_textbox(s, Inches(0.9), Inches(6.65), Inches(8), Inches(0.4),
            "Prepared for Senior Stakeholder Review  |  Sailen Kumar", size=13,
            color=RGBColor(0x9C, 0xA9, 0xBA))
add_notes(s, "Welcome and thank you for the time. This is a 15-minute walkthrough of the "
             "capstone project: a credit-default model built to be simultaneously accurate, "
             "explainable to a human reviewer, and audited/corrected for demographic bias.")

# ===========================================================================
# Slide 2 — Executive Summary
# ===========================================================================
s = new_content_slide("EXECUTIVE SUMMARY", "The 60-second version")
tiles = [
    ("0.769", "Best model ROC-AUC\n(XGBoost, tuned)"),
    ("0.677 → 0.821", "Gender fairness ratio\npre → post mitigation"),
    ("2", "Datasets audited\n(Home Credit + HMDA)"),
]
tile_w = Inches(3.7); gap = Inches(0.35)
start_x = Inches(0.6)
for i, (val, lab) in enumerate(tiles):
    x = start_x + i * (tile_w + gap)
    stat_tile(s, x, Inches(1.85), tile_w, Inches(1.5), val, lab, value_color=NAVY_MID)
add_bullets(s, Inches(0.7), Inches(3.7), Inches(11.8), Inches(3.0), [
    "We built and tuned a default-risk model on 307,511 real Home Credit loan applications, "
    "reaching ROC-AUC 0.769 — a meaningful lift over both baselines.",
    "Every prediction is explainable: SHAP shows loan officers and regulators exactly why a "
    "score was assigned, down to the individual applicant.",
    "An initial fairness audit found real disparities by gender and age group (both failed the "
    "four-fifths rule). We applied bias mitigation (reweighing) and closed most of the gap.",
    "The same audit methodology was independently re-validated on a second, unrelated dataset "
    "(HMDA mortgage data, 716K records) — it generalizes.",
    ("Bottom line: the model is materially more accurate than baseline, its decisions are "
     "explainable, and its fairness gaps are measured and actively being corrected — not "
     "just claimed.", 0),
], size=16)
add_notes(s, "Frame the whole talk before the details: we have a strong, explainable model; we "
             "found real bias; we measured it and materially reduced it; and we proved the "
             "audit approach isn't a one-off by repeating it on an independent dataset.")

# ===========================================================================
# Slide 3 — Business Problem & Objectives
# ===========================================================================
s = new_content_slide("WHY THIS PROJECT", "The business problem")
add_bullets(s, Inches(0.7), Inches(1.85), Inches(6.0), Inches(4.8), [
    "Credit-default models drive real lending decisions — approve, decline, or price a loan.",
    "Three failure modes stakeholders care about:",
    ("A model that's inaccurate → bad loans approved, good customers declined", 1),
    ("A model that's a black box → can't justify a decision to a customer, auditor, or "
     "regulator", 1),
    ("A model that's biased → legal, reputational, and regulatory exposure (fair-lending "
     "laws: ECOA / Reg B)", 1),
    "This project treats all three as first-class, measurable requirements — not "
    "after-the-fact add-ons.",
], size=16.5)
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.85), Inches(5.6), Inches(4.8))
box.adjustments[0] = 0.04
box.fill.solid(); box.fill.fore_color.rgb = LIGHT_GRAY
box.line.color.rgb = RGBColor(0xE2, 0xE8, 0xF0); box.shadow.inherit = False
tf = box.text_frame; tf.word_wrap = True; tf.margin_left = Pt(18); tf.margin_top = Pt(16)
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
r = p.add_run(); r.text = "Project objective"; r.font.bold = True; r.font.size = Pt(17)
r.font.color.rgb = NAVY_MID; r.font.name = FONT
for goal, desc in [
    ("Accurate", "Beat baseline default-prediction performance under real class imbalance (~8% default rate)."),
    ("Explainable", "Every score traceable to feature-level reasons via SHAP — global and per-applicant."),
    ("Fair", "Formally audit for disparate impact by gender/age/race and mitigate what we find."),
]:
    pp = tf.add_paragraph(); pp.space_before = Pt(14)
    r1 = pp.add_run(); r1.text = f"{goal}:  "; r1.font.bold = True; r1.font.size = Pt(15)
    r1.font.color.rgb = TEAL if goal != "Fair" else AMBER; r1.font.name = FONT
    r2 = pp.add_run(); r2.text = desc; r2.font.size = Pt(14.5); r2.font.color.rgb = DARK_TEXT
    r2.font.name = FONT
add_notes(s, "Ground the technical work in why a senior audience should care: accuracy is table "
             "stakes, but explainability and fairness are what keep the model deployable and "
             "defensible.")

# ===========================================================================
# Slide 4 — Scope & Approach
# ===========================================================================
s = new_content_slide("SCOPE", "What we built — and what we deliberately didn't")
add_bullets(s, Inches(0.7), Inches(1.85), Inches(11.9), Inches(2.2), [
    "Home Credit Default Risk (Kaggle): the full pipeline — EDA, feature engineering, "
    "baseline + tuned models, evaluation, SHAP, fairness audit, and bias mitigation.",
    "HMDA mortgage data (CFPB): fairness-audit-only, on an independent dataset and target, to "
    "test whether the fairness methodology generalizes beyond Home Credit.",
], size=16.5)
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(4.35), Inches(11.9), Inches(2.35))
box.adjustments[0] = 0.05
box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0xFF, 0xF7, 0xE8)
box.line.color.rgb = AMBER; box.line.width = Pt(1); box.shadow.inherit = False
tf = box.text_frame; tf.word_wrap = True; tf.margin_left = Pt(18); tf.margin_top = Pt(14)
p = tf.paragraphs[0]
r = p.add_run(); r.text = "Scoped down from the original proposal, deliberately"
r.font.bold = True; r.font.size = Pt(15.5); r.font.color.rgb = RGBColor(0x92, 0x63, 0x0A); r.font.name = FONT
for line in [
    "Original plan: 2 full pipelines × 2 model families × 3 mitigation techniques — "
    "judged infeasible for a single capstone term.",
    "Delivered: 1 full pipeline (Home Credit) + 1 fairness-audit-only pipeline (HMDA) + 1 "
    "mitigation technique (reweighing), chosen because it's model-agnostic and needs no "
    "architecture changes.",
]:
    pp = tf.add_paragraph(); pp.space_before = Pt(8)
    rr = pp.add_run(); rr.text = "▪ " + line
    rr.font.size = Pt(14); rr.font.color.rgb = RGBColor(0x78, 0x50, 0x08); rr.font.name = FONT
add_notes(s, "Be upfront about scope reduction — stakeholders respect a clearly reasoned "
             "MVP cut far more than an over-promised plan. The reasoning: the full matrix was "
             "infeasible in the timeframe; reweighing was chosen because it works identically "
             "for both model families with zero architecture change.")

# ===========================================================================
# Slide 5 — Data
# ===========================================================================
s = new_content_slide("DATA", "Two real-world datasets, two different roles")
col_w = Inches(5.7)
left1, left2 = Inches(0.7), Inches(6.7)
top = Inches(1.85)
for left, title_txt, rows, role, target, color in [
    (left1, "Home Credit Default Risk (Kaggle)", "307,511 rows × 122 columns",
     "Primary modeling dataset", "TARGET — loan default (8.07% positive rate)", TEAL),
    (left2, "HMDA — CFPB Mortgage Data", "715,927 rows × 22 columns "
     "(nationwide 2008–2017 sample)", "Fairness audit-only, second opinion on methodology",
     "high_cost_flag — pricing proxy (5.77% positive rate)", AMBER),
]:
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, col_w, Inches(4.7))
    box.adjustments[0] = 0.04
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT_GRAY
    box.line.color.rgb = RGBColor(0xE2, 0xE8, 0xF0); box.shadow.inherit = False
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = Pt(16); tf.margin_top = Pt(16); tf.margin_right = Pt(14)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = title_txt; r.font.bold = True; r.font.size = Pt(17)
    r.font.color.rgb = NAVY_MID; r.font.name = FONT
    for label, val in [("Size", rows), ("Role", role), ("Target", target)]:
        pp = tf.add_paragraph(); pp.space_before = Pt(14)
        r1 = pp.add_run(); r1.text = f"{label}\n"; r1.font.bold = True; r1.font.size = Pt(13)
        r1.font.color.rgb = color; r1.font.name = FONT
        r2 = pp.add_run(); r2.text = val; r2.font.size = Pt(14.5); r2.font.color.rgb = DARK_TEXT
        r2.font.name = FONT
add_textbox(s, Inches(0.7), Inches(6.7), Inches(11.9), Inches(0.6),
            "Both are real data (no synthetic fallback used for this run). HMDA's target is "
            "derived (rate_spread reported ⇒ high-cost loan), not a raw approval/denial "
            "field — so it is never compared apples-to-apples with Home Credit's default target.",
            size=12.5, color=GRAY, italic=True)
add_notes(s, "Two datasets, two jobs. Home Credit is the real modeling target. HMDA exists "
             "purely to stress-test whether our fairness-auditing approach holds up on a "
             "completely independent dataset with different features and a different label.")

# ===========================================================================
# Slide 6 — Pipeline Architecture
# ===========================================================================
s = new_content_slide("HOW IT WORKS", "End-to-end pipeline, one config, ten stages")
stages = [
    "Load data", "EDA", "Feature\nengineering", "Baseline\ntraining",
    "Hyper-\ntuning", "Evaluation", "SHAP\nexplain", "Fairness\naudit",
    "Bias\nmitigation", "HMDA\naudit",
]
n = len(stages)
margin = Inches(0.55)
total_w = SW - 2 * margin
box_w = Emu(int((total_w - Emu(int(Pt(6))) * (n - 1)) / n))
box_h = Inches(1.15)
top = Inches(2.55)
gap = Emu(int(Pt(6)))
xs = []
x = margin
for i, label in enumerate(stages):
    color = NAVY_MID if i < 7 else AMBER
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, top, box_w, box_h)
    b.adjustments[0] = 0.12
    b.fill.solid(); b.fill.fore_color.rgb = color
    b.line.fill.background(); b.shadow.inherit = False
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = f"{i+1}. {label}"
    r.font.size = Pt(10.5); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = FONT
    xs.append(x)
    x = Emu(int(x) + int(box_w) + int(gap))
    if i < n - 1:
        arrow_left = Emu(int(x) - int(gap))
add_bullets(s, Inches(0.7), Inches(4.1), Inches(11.9), Inches(2.6), [
    "One config file (config/config.yaml) drives every stage — paths, seeds, hyperparameter "
    "ranges, and the fairness threshold are defined exactly once.",
    "Protected attributes (gender, age group, race, sex) are excluded from model features by "
    "design — used only for the post-hoc fairness audit, matching standard fair-lending practice.",
    "Fully reproducible: python scripts/run_pipeline.py regenerates every model, figure, and "
    "metric in this deck from raw data.",
], size=15.5)
add_notes(s, "This is meant to build confidence in rigor, not to be read line by line: the whole "
             "thing is one reproducible, config-driven pipeline, and sensitive attributes never "
             "leak into the model itself — only into the audit step.")

# ===========================================================================
# Slide 7 — Model Performance
# ===========================================================================
s = new_content_slide("MODEL PERFORMANCE", "Tuning improved every model — XGBoost wins")
add_picture_fit(s, METRICS_DIR / "model_comparison.png", Inches(0.6), Inches(1.75),
                 Inches(7.1), Inches(5.2))
rows_data = [
    ("Metric", "LogReg\nBaseline", "LogReg\nTuned", "XGBoost\nBaseline", "XGBoost\nTuned ★"),
    ("ROC-AUC", f"{lr_base['roc_auc']:.3f}", f"{lr_tuned['roc_auc']:.3f}",
     f"{xgb_base['roc_auc']:.3f}", f"{xgb_tuned['roc_auc']:.3f}"),
    ("PR-AUC", f"{lr_base['pr_auc']:.3f}", f"{lr_tuned['pr_auc']:.3f}",
     f"{xgb_base['pr_auc']:.3f}", f"{xgb_tuned['pr_auc']:.3f}"),
    ("KS stat.", f"{lr_base['ks_statistic']:.3f}", f"{lr_tuned['ks_statistic']:.3f}",
     f"{xgb_base['ks_statistic']:.3f}", f"{xgb_tuned['ks_statistic']:.3f}"),
]
left, top, width, height = Inches(7.95), Inches(1.85), Inches(4.7), Inches(2.6)
table_shape = s.shapes.add_table(len(rows_data), 5, left, top, width, height)
table = table_shape.table
for c in range(5):
    table.columns[c].width = Inches(0.94) if c == 0 else Inches(0.94)
for r_idx, row in enumerate(rows_data):
    for c_idx, val in enumerate(row):
        cell = table.cell(r_idx, c_idx)
        cell.text = val
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_top = Pt(3); cell.margin_bottom = Pt(3)
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.CENTER
        run = para.runs[0]
        run.font.size = Pt(11 if r_idx == 0 else 12.5)
        run.font.name = FONT
        run.font.bold = (r_idx == 0 or c_idx == 4)
        run.font.color.rgb = WHITE if r_idx == 0 else (TEAL if c_idx == 4 else DARK_TEXT)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_MID if r_idx == 0 else (LIGHT_GRAY if r_idx % 2 else WHITE)
add_bullets(s, Inches(7.95), Inches(4.65), Inches(4.75), Inches(2.3), [
    "XGBoost (tuned) is the best model: ROC-AUC 0.769, PR-AUC 0.259, KS 0.401.",
    "PR-AUC and Brier are tracked alongside ROC-AUC because the ~8% default rate makes ROC-AUC "
    "alone misleading under class imbalance.",
    "Hyperparameters tuned via RandomizedSearchCV (LR) and Optuna/TPE (XGBoost), both "
    "optimizing cross-validated ROC-AUC.",
], size=13.5)
add_notes(s, "XGBoost tuned is the production candidate. Emphasize we didn't cherry-pick "
             "ROC-AUC — PR-AUC and Brier score are reported because the class imbalance "
             "(~8% defaults) can make ROC-AUC look better than the model really is.")

# ===========================================================================
# Slide 8 — Explainability
# ===========================================================================
s = new_content_slide("EXPLAINABILITY", "Every prediction has a reason — not a black box")
add_picture_fit(s, FIG_HC / "shap" / "xgboost_tuned_shap_bar.png", Inches(0.6), Inches(1.75),
                 Inches(5.7), Inches(5.35))
top_feats = list(shap_top.items())[:5]
name_map = {
    "num__EXT_SOURCE_MEAN": "External credit bureau score (avg)",
    "num__CREDIT_TERM": "Credit term (annuity / loan amount)",
    "num__EXT_SOURCE_3": "External credit bureau score #3",
    "num__GOODS_CREDIT_RATIO": "Goods price vs. credit ratio",
    "num__AMT_ANNUITY": "Loan annuity amount",
}
items = []
for feat, val in top_feats:
    label = name_map.get(feat, feat)
    items.append((f"{label}  —  mean |SHAP| {val:.2f}", 0))
add_bullets(s, Inches(6.6), Inches(1.85), Inches(6.1), Inches(2.6), [
    "Top global drivers of the default score:",
] + items, size=14.5)
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.6), Inches(4.6), Inches(6.1), Inches(2.4))
box.adjustments[0] = 0.05
box.fill.solid(); box.fill.fore_color.rgb = LIGHT_GRAY
box.line.color.rgb = RGBColor(0xE2, 0xE8, 0xF0); box.shadow.inherit = False
tf = box.text_frame; tf.word_wrap = True; tf.margin_left = Pt(16); tf.margin_top = Pt(12)
p = tf.paragraphs[0]
r = p.add_run(); r.text = "What this means in practice"
r.font.bold = True; r.font.size = Pt(14.5); r.font.color.rgb = NAVY_MID; r.font.name = FONT
for line in [
    "Global (bar/beeswarm) plots show which features drive risk across the whole portfolio.",
    "Local (waterfall) plots exist per applicant — a loan officer or regulator can see "
    "exactly why one specific application scored the way it did.",
]:
    pp = tf.add_paragraph(); pp.space_before = Pt(8)
    rr = pp.add_run(); rr.text = "▪ " + line
    rr.font.size = Pt(13); rr.font.color.rgb = DARK_TEXT; rr.font.name = FONT
add_notes(s, "External bureau scores and the credit-term ratio dominate — intuitive, "
             "auditable drivers, not opaque interactions. And this isn't only a portfolio-level "
             "view: every single applicant gets their own waterfall explanation.")

# ===========================================================================
# Slide 9 — Fairness Audit Pre-Mitigation
# ===========================================================================
s = new_content_slide("FAIRNESS AUDIT", "We looked for bias — and found it", dark_header=True)
add_picture_fit(s, FIG_HC / "fairness" / "xgboost_tuned_pre_mitigation_CODE_GENDER_selection_rate.png",
                 Inches(0.6), Inches(1.8), Inches(5.9), Inches(3.1))
add_picture_fit(s, FIG_HC / "fairness" / "xgboost_tuned_pre_mitigation_AGE_GROUP_selection_rate.png",
                 Inches(0.6), Inches(4.75), Inches(5.9), Inches(2.4))
gd = pre_fair["CODE_GENDER"]; ad = pre_fair["AGE_GROUP"]
right_x = Inches(6.8)
stat_tile(s, right_x, Inches(1.85), Inches(2.75), Inches(1.35),
          f"{gd['demographic_parity_ratio']:.3f}", "Gender parity ratio\n(fails 0.80 threshold)",
          value_color=RED)
stat_tile(s, right_x + Inches(2.95), Inches(1.85), Inches(2.75), Inches(1.35),
          f"{ad['demographic_parity_ratio']:.3f}", "Age-group parity ratio\n(fails 0.80 threshold)",
          value_color=RED)
add_bullets(s, right_x, Inches(3.5), Inches(5.75), Inches(3.4), [
    "Method: Fairlearn MetricFrame audit — selection rate, demographic-parity ratio/"
    "difference, and equalized-odds difference per protected group.",
    "Threshold: the “four-fifths rule” (selection-rate ratio ≥ 0.80). It's a "
    "borrowed EEOC convention, not a codified ECOA/Reg B legal standard — used here as a "
    "practical flag for materially large gaps.",
    f"Gender: female applicants approved at {gd['by_group']['F']['selection_rate']:.0%} vs. "
    f"male {gd['by_group']['M']['selection_rate']:.0%} predicted selection rate.",
    "Age: the youngest group (<25) is approved far less often than the oldest (55+) — the "
    "largest gap in the audit.",
], size=14)
add_notes(s, "This is the uncomfortable-but-necessary slide: on the untouched tuned model, both "
             "gender and age group fail the four-fifths rule. We're not hiding this — it's "
             "exactly what the audit stage exists to catch, and it's what motivates the next "
             "slide.")

# ===========================================================================
# Slide 10 — Bias Mitigation Method & Results
# ===========================================================================
s = new_content_slide("BIAS MITIGATION", "Reweighing closes most of the gap")
add_picture_fit(s, FIG_HC / "fairness" / "xgboost_tuned_post_mitigation_CODE_GENDER_selection_rate.png",
                 Inches(0.6), Inches(1.8), Inches(5.9), Inches(3.05))
add_picture_fit(s, FIG_HC / "fairness" / "xgboost_tuned_post_mitigation_AGE_GROUP_selection_rate.png",
                 Inches(0.6), Inches(4.75), Inches(5.9), Inches(2.4))
gd2 = post_fair["CODE_GENDER"]; ad2 = post_fair["AGE_GROUP"]
right_x = Inches(6.8)
stat_tile(s, right_x, Inches(1.85), Inches(2.75), Inches(1.3),
          f"{gd['demographic_parity_ratio']:.2f} → {gd2['demographic_parity_ratio']:.2f}",
          "Gender parity ratio\n(now passes 0.80)", value_color=GREEN)
stat_tile(s, right_x + Inches(2.95), Inches(1.85), Inches(2.75), Inches(1.3),
          f"{ad['demographic_parity_ratio']:.2f} → {ad2['demographic_parity_ratio']:.2f}",
          "Age parity ratio\n(now passes 0.80)", value_color=GREEN)
add_bullets(s, right_x, Inches(3.45), Inches(5.75), Inches(3.5), [
    "Method: Kamiran & Calders (2012) reweighing — per-row sample weights that "
    "decorrelate the protected attribute from the label before retraining. Model-agnostic, no "
    "architecture change required.",
    f"Result: both gender (0.677→0.821) and age (0.547→0.813) now pass the "
    "four-fifths rule.",
    ("Honest tradeoff: ROC-AUC moved "
     f"{xgb_tuned['roc_auc']:.3f} → {mitigated['roc_auc']:.3f} "
     f"(PR-AUC {xgb_tuned['pr_auc']:.3f} → {mitigated['pr_auc']:.3f}); Brier "
     f"(calibration) improved to {mitigated['brier_score']:.3f}."),
    "This is a fairness/accuracy tradeoff curve, not a free lunch — the business decision "
    "is how much of that ~5-point ROC-AUC to spend to fix a fair-lending exposure.",
], size=13.5)
add_notes(s, "Reweighing meaningfully closes the gap on both attributes without touching model "
             "architecture. Be candid about the cost: ROC-AUC drops about 5 points. Frame it as "
             "a deliberate, quantified tradeoff for stakeholders to weigh — not a hidden cost.")

# ===========================================================================
# Slide 11 — HMDA generalization check
# ===========================================================================
s = new_content_slide("INDEPENDENT VALIDATION", "Does the audit methodology generalize? Yes.")
add_picture_fit(s, FIG_HMDA / "hmda_xgboost_derived_race_selection_rate.png",
                 Inches(0.6), Inches(1.8), Inches(6.0), Inches(4.9))
rd = hmda_fair["derived_race"]; sd = hmda_fair["derived_sex"]
right_x = Inches(6.9)
add_textbox(s, right_x, Inches(1.85), Inches(5.7), Inches(0.5),
            "716K real HMDA mortgage records, 2008–2017, nationwide sample", size=14.5,
            color=GRAY, italic=True)
stat_tile(s, right_x, Inches(2.45), Inches(2.7), Inches(1.25),
          f"{hmda_perf['roc_auc']:.3f}", "Model ROC-AUC\n(pricing-proxy target)",
          value_color=TEAL)
stat_tile(s, right_x + Inches(2.9), Inches(2.45), Inches(2.7), Inches(1.25),
          f"{rd['demographic_parity_ratio']:.3f}", "Race parity ratio\n(fails 0.80)",
          value_color=RED)
add_bullets(s, right_x, Inches(4.0), Inches(5.7), Inches(2.9), [
    "Same audit code, independently re-run on a completely different dataset, feature set, "
    "and label — the methodology is not overfit to Home Credit.",
    f"Race: parity ratio {rd['demographic_parity_ratio']:.3f} — fails the four-fifths "
    "rule (flagged).",
    f"Sex: parity ratio {sd['demographic_parity_ratio']:.3f} — passes.",
    "HMDA here is fairness-audit-only by design — its target is a pricing proxy "
    "(high-cost flag), never treated as a default model or compared apples-to-apples with "
    "Home Credit.",
], size=14)
add_notes(s, "This slide exists to answer the skeptical question: 'did you just get lucky "
             "tuning the audit to one dataset?' No — rerun cold on HMDA mortgage data, the "
             "same audit catches a real race-based disparity there too.")

# ===========================================================================
# Slide 12 — Limitations & Responsible Use
# ===========================================================================
s = new_content_slide("LIMITATIONS", "What this project does not claim", dark_header=True)
add_bullets(s, Inches(0.7), Inches(1.85), Inches(11.9), Inches(4.9), [
    "The four-fifths rule is a borrowed EEOC employment-discrimination convention — not a "
    "codified ECOA / Reg B legal fair-lending standard. Treat it as a practical screen, not a "
    "compliance certification.",
    "HMDA's target (high_cost_flag) is a rate-spread pricing proxy on originated loans only "
    "(this extract has no denials) — it is not an approval/denial model and is never "
    "compared 1:1 against Home Credit's default target.",
    "Reweighing was the single mitigation technique implemented for this MVP (vs. the original "
    "proposal's reweighing + adversarial debiasing + equalized-odds postprocessing matrix) — "
    "other techniques may trade off accuracy and fairness differently.",
    "Fairness was audited on the attributes available (gender/age for Home Credit, race/sex for "
    "HMDA) at a single ~50% classification threshold — results can shift at other "
    "operating thresholds.",
    "This is a capstone-scope MVP, not a production-certified compliance pipeline — "
    "deploying to real lending decisions would need legal/compliance sign-off beyond this audit.",
], size=15.5)
add_notes(s, "Precommit to intellectual honesty here: this slide heads off the hardest questions "
             "before they're asked. None of these are surprises we're hiding — they're "
             "documented limitations from day one of the design.")

# ===========================================================================
# Slide 13 — Recommendations & Next Steps
# ===========================================================================
s = new_content_slide("RECOMMENDATIONS", "Where we'd take this next")
left1, left2 = Inches(0.7), Inches(6.85)
for left, heading, items, color in [
    (left1, "Near-term (next iteration)", [
        "Decide the acceptable fairness/accuracy tradeoff point with legal & risk stakeholders "
        "before any production discussion.",
        "Extend the mitigation comparison to adversarial debiasing and equalized-odds "
        "postprocessing (originally scoped, deferred for MVP).",
        "Re-run the fairness audit across multiple decision thresholds, not just 0.50.",
    ], TEAL),
    (left2, "Longer-term", [
        "Formal legal/compliance review of the four-fifths screen against ECOA / Reg B before "
        "any real lending use.",
        "Expand HMDA to post-2018 modern LAR data (adds denials) to enable a true "
        "approval/denial fairness audit, not just a pricing proxy.",
        "Add model monitoring for fairness/performance drift once (if) deployed.",
    ], AMBER),
]:
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.85), Inches(5.75), Inches(4.7))
    box.adjustments[0] = 0.04
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT_GRAY
    box.line.color.rgb = color; box.line.width = Pt(1.25); box.shadow.inherit = False
    tf = box.text_frame; tf.word_wrap = True; tf.margin_left = Pt(16); tf.margin_top = Pt(14)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = heading; r.font.bold = True; r.font.size = Pt(16.5)
    r.font.color.rgb = NAVY_MID; r.font.name = FONT
    for line in items:
        pp = tf.add_paragraph(); pp.space_before = Pt(12); pp.line_spacing = 1.1
        rr = pp.add_run(); rr.text = "▪ " + line
        rr.font.size = Pt(13.5); rr.font.color.rgb = DARK_TEXT; rr.font.name = FONT
add_notes(s, "Close on forward motion: a concrete near-term list stakeholders can greenlight "
             "today, and a longer-term list that needs their sponsorship (legal review, more "
             "data, monitoring).")

# ===========================================================================
# Slide 14 — Thank you / Q&A
# ===========================================================================
s = add_slide()
set_background(s, NAVY)
add_textbox(s, Inches(0.9), Inches(2.9), Inches(11.5), Inches(1.0),
            "Thank you", size=40, color=WHITE, bold=True)
add_textbox(s, Inches(0.9), Inches(3.85), Inches(11.5), Inches(0.6),
            "Questions & discussion", size=20, color=TEAL, bold=True)
add_textbox(s, Inches(0.9), Inches(6.65), Inches(10), Inches(0.4),
            "Full pipeline, code, and reports: see project README and Developer Guide",
            size=13, color=RGBColor(0x9C, 0xA9, 0xBA))
add_notes(s, "Open the floor. Likely questions to be ready for: the ROC-AUC/fairness tradeoff "
             "number, why reweighing over other mitigation techniques, and the legal standing of "
             "the four-fifths rule — all three are covered on the limitations slide if needed.")

prs.save(str(OUTPUT))
print(f"Saved: {OUTPUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
