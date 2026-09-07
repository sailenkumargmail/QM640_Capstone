# Credit Default Prediction: Explainable AI & Bias Mitigation

Capstone project (QM640, Walsh College) implementing a credit-default
scoring pipeline that is simultaneously (1) **accurate**, (2)
**explainable**, and (3) **fairness-audited and bias-mitigated**, against
the **UCI "default of credit card clients"** dataset (Yeh & Lien, 2009;
30,000 real Taiwanese credit-card accounts).

**See [docs/DEVELOPER_GUIDE.pdf](docs/DEVELOPER_GUIDE.pdf) (or the
[Markdown source](docs/DEVELOPER_GUIDE.md)) for the full high-level design,
setup, and running guide.**

## Overview

Four model families are trained, tuned, and compared:
- **Logistic Regression** — interpretable linear baseline.
- **XGBoost** and **CatBoost** — two independently implemented gradient-
  boosted tree ensembles, included together so any accuracy gain isn't one
  library's artifact.
- **EBM** (Explainable Boosting Machine) — a glass-box GAM, natively
  interpretable rather than needing a post-hoc explainer.

The best-performing model (by held-out ROC-AUC) is explained with SHAP,
audited for fairness on `SEX` and a derived `AGE_GROUP`, and re-trained with
two independent mitigation techniques — **reweighing** (pre-training) and
**equalized-odds post-processing** (Fairlearn `ThresholdOptimizer`,
post-training). The same reweighing recipe is then replicated, unmodified,
across all four model families to test whether it generalizes beyond the
single best model.

Four research questions structure the project:
- **RQ1 (Accuracy):** Do tuned XGBoost/CatBoost/EBM beat tuned Logistic
  Regression on ROC-AUC?
- **RQ2 (Explainability):** Do SHAP and EBM's native feature-importance
  rankings agree with each other?
- **RQ3 (Fairness):** Does the best model show demographic-parity gaps on
  `SEX`/`AGE_GROUP`, and does reweighing fix them?
- **RQ4 (Generalization):** Does that reweighing recipe work comparably
  across all four model families?

> **Legacy scope note:** an earlier iteration of this project used the
> Home Credit Default Risk (Kaggle) and HMDA (CFPB mortgage) datasets. That
> scope was superseded by a pivot to the UCI dataset above. The old
> download/prep scripts (`scripts/download_home_credit.py`,
> `scripts/download_hmda.py`, `scripts/prepare_hmda_real.py`) and the
> `home_credit`/`hmda` folders under `data/` are kept for provenance but are
> **not** part of the current pipeline (`config/config.yaml` and
> `scripts/run_pipeline.py` are UCI-only).

## Data

The pipeline runs against real data by default. The raw UCI source export
ships in the repo at `data/external/UCI/default of credit card clients.xls`
and is prepared into a clean CSV by:

```bash
python scripts/prepare_uci_credit.py
```

This renames the target column to `DEFAULT_PAYMENT_NEXT_MONTH`, maps `SEX`
to `Male`/`Female`, and collapses undocumented `EDUCATION`/`MARRIAGE` codes
into `"Other/Unknown"`, writing
`data/raw/uci_credit/default_of_credit_card_clients.csv`.

If that prepared file is absent, `dac.data.loader.load_uci_credit()`
transparently falls back to a schema-accurate **synthetic dataset**
(`src/dac/data/synthetic.py`) with a small, documented synthetic bias term
injected on `SEX`/`AGE_GROUP`, so the fairness-audit/mitigation stages
always have a real, known effect to detect and correct — no code changes
needed either way, and it's cached as a parquet file so repeated runs don't
regenerate it.

See [data/README.md](data/README.md) for the full expected layout.

## Project structure

```
QM640_Capstone/
├── config/config.yaml          # single source of truth: paths, seed, model/tuning/fairness params
├── data/{raw,processed,external}  # gitignored; see data/README.md
├── docs/                        # Developer Guide, stakeholder deck, synopsis doc
├── models/                      # trained model artifacts (joblib), gitignored
├── notebooks/                   # 01-05, one per pipeline stage group
├── reports/{figures,model_performance}  # generated EDA/eval/fairness/significance output
├── scripts/
│   ├── prepare_uci_credit.py       # real-data preparation (xls -> csv, cleanup)
│   ├── run_pipeline.py             # end-to-end orchestrator (main entry point, ~39 min)
│   ├── run_extended_analysis.py    # hypothesis-test p-values, 2nd mitigation technique, decision framework (<2 min)
│   ├── build_rq_flowchart.py       # RQ1-RQ4 lineage / solution-flow diagram
│   ├── build_interim_report_uci.py # Interim Report docx generator
│   ├── build_final_report.py       # Final Report docx generator
│   ├── build_stakeholder_ppt.py    # stakeholder summary deck generator
│   └── build_report_evidence.py    # data-preview/repo-tree screenshots for reports
├── src/dac/
│   ├── config.py
│   ├── data/                   # loading (load_uci_credit) + synthetic data generation
│   ├── features/                # feature engineering + preprocessing
│   ├── eda/                     # EDA report generation
│   ├── models/                  # train / tune / evaluate (LR, XGBoost, CatBoost, EBM)
│   ├── explainability/          # SHAP + EBM native explanation, SHAP-vs-EBM agreement
│   └── fairness/                # audit (Fairlearn), mitigation (reweighing), postprocessing
│       └── significance.py      #   bootstrap AUC test, two-proportion z-tests
└── tests/                       # pytest unit + integration smoke tests
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
pip install -e .                  # installs the `dac` package from src/
```

## Running the pipeline

```bash
python scripts/prepare_uci_credit.py      # once, or whenever the source .xls changes
python scripts/run_pipeline.py            # full run (~39 min, dominated by Optuna tuning)
python scripts/run_pipeline.py --quick    # fast smoke-test run (fewer tuning trials)
```

This runs, in order: EDA → feature engineering → baseline training →
hyperparameter tuning (RandomizedSearchCV for LR, Optuna for
XGBoost/CatBoost/EBM) → evaluation (ROC-AUC, PR-AUC, F1, KS, Brier) → SHAP
explainability + EBM native explanation (RQ2 agreement) → fairness audit on
`SEX`/`AGE_GROUP` (RQ3) → reweighing mitigation + re-audit for all four
models (RQ4). Figures land in `reports/figures/uci_credit/`, metrics/tables
in `reports/model_performance/` (including a consolidated
`run_summary.json`), and trained models in `models/`.

To close out the exact hypothesis-test p-values, a joint `SEX x AGE_GROUP`
mitigation sensitivity run, a second bias-mitigation technique
(equalized-odds post-processing), and an accuracy-fairness decision
framework — without re-running the ~39-minute tuning step:

```bash
python scripts/run_extended_analysis.py   # reloads models/*.joblib, runs in <2 min
```

This writes `reports/model_performance/extended_summary.json` plus the
`significance/`, `sensitivity/`, `mitigation2_equalized_odds/`, and
`decision_framework.*` outputs under `reports/model_performance/`.

## Report deliverables

The Interim Report, Final Report, and Final Presentation deliverables are
generated by the scripts above but live **outside this repo**, alongside the
university's structural templates — see the paths in
`scripts/build_final_report.py` and `scripts/build_interim_report_uci.py`
for what each one produces.

## Tests

```bash
pytest
```

`tests/test_pipeline.py` runs the entire pipeline end-to-end on a tiny
synthetic sample as an integration smoke test; the rest are unit tests per
module (`test_data.py`, `test_features.py`, `test_models.py`,
`test_fairness.py`).

## Methodology notes / known limitations

- **Four-fifths rule**: the 0.80 disparate-impact-ratio threshold used in
  `dac.fairness.audit` is an EEOC employment-discrimination heuristic, not a
  codified lending-fairness standard under ECOA/Reg B. It's used here as a
  common fairness-ML convention, not a claimed regulatory threshold. The
  project's own results show why a ratio-only check is insufficient: on the
  best model, `AGE_GROUP` clears the 0.80 ratio pre-mitigation yet still
  fails a two-proportion z-test for a statistically significant gap.
- **Two mitigation techniques** are compared: reweighing (Kamiran & Calders,
  2012; pre-training, model-agnostic) as the primary technique, and
  Fairlearn's `ThresholdOptimizer` (equalized-odds post-processing) as a
  second, independent technique for the best model — added via
  `scripts/run_extended_analysis.py` to test whether the reweighing result
  holds under a different mitigation approach.
- **PR-AUC, KS statistic, and Brier score** are reported alongside
  ROC-AUC/F1 because the target's ~22% positive rate makes ROC-AUC alone
  potentially misleading under class imbalance.
- Protected attributes (`SEX`, derived `AGE_GROUP`) are **excluded from
  model features** and used only for post-hoc fairness auditing, matching
  standard fair-lending practice.
- **Single-dataset evaluation**: results come from one Taiwanese dataset
  from one time period (2005); generalization to other geographies or more
  recent lending cohorts is untested (see Conclusion / Future Work in the
  Final Report).
