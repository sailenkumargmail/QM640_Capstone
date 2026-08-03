# Developer Guide

**Credit Default Prediction: Explainable AI & Bias Mitigation**
QM640 Capstone Project

---

## Contents

1. Overview
2. High-Level Design
3. Project Structure
4. Setup Guide
5. Running Guide
6. Interpreting the Outputs
7. Extending the Project
8. Troubleshooting
9. Appendix: Configuration Reference

---

## 1. Overview

This project implements a credit-default scoring pipeline that is
simultaneously **accurate**, **explainable**, and **fairness-audited**,
against the **UCI "default of credit card clients"** dataset (Yeh & Lien,
2009; 30,000 Taiwanese credit-card accounts, target =
`DEFAULT_PAYMENT_NEXT_MONTH`).

Four model families are trained and compared: **Logistic Regression**
(interpretable linear baseline), **XGBoost** and **CatBoost** (gradient-
boosted tree ensembles), and **Explainable Boosting Machine (EBM)** — a
glass-box generalized additive model that is *natively* interpretable rather
than needing a post-hoc explainer. That last point drives one of the
project's four research questions: does EBM's own, exact explanation agree
with SHAP's post-hoc approximation on the same kind of model?

The best-performing model (by held-out ROC-AUC) is explained with SHAP,
audited for fairness on `SEX` and `AGE_GROUP`, and re-trained with reweighing
mitigation; the same mitigation recipe is then replicated, unmodified, across
all four model families to test whether it generalizes beyond the single
best model.

### Data availability

The real UCI extract ships in the repo at `data/external/UCI/default of
credit card clients.xls` and is prepared into a clean CSV by
`scripts/prepare_uci_credit.py` — no external account or API credentials are
needed. A schema-accurate synthetic generator (`src/dac/data/synthetic.py`)
is available as a fallback if that preparation step hasn't been run (e.g. for
fast unit tests), with a small, explicitly documented synthetic bias term
injected on `SEX`/`AGE_GROUP` so the fairness-audit and mitigation stages
have a real, known effect to detect and correct.

---

## 2. High-Level Design

### 2.1 Pipeline architecture

The system is a linear, nine-stage pipeline orchestrated by
`scripts/run_pipeline.py`. Every stage is also independently importable from
the `dac` package.

```
 1. Load data           dac.data.loader              (real file if present, else synthetic)
 2. EDA                 dac.eda.eda_report            -> reports/figures/, eda_uci_credit.md
 3. Feature engineering dac.features.engineering      -> engineered DataFrame + train/test split
 4. Baseline training   dac.models.train               -> LR, XGBoost, CatBoost, EBM
 5. Hyperparameter tuning dac.models.tune              -> RandomizedSearchCV (LR), Optuna/TPE (XGB/CatBoost/EBM)
 6. Evaluation          dac.models.evaluate            -> ROC-AUC, PR-AUC, F1, KS, Brier + plots; best-model selection
 7. Explainability      dac.explainability.shap_explain / ebm_explain -> SHAP (XGBoost + best model) + EBM native explanation + agreement
 8. Fairness audit      dac.fairness.audit             -> Fairlearn group metrics, all 4 tuned models (pre-mitigation)
 9. Bias mitigation     dac.fairness.mitigation        -> reweighing on SEX, retrain + re-audit, all 4 models (post-mitigation)
```

Every stage reads from and writes to a single `CONFIG` dict
(`dac.config.CONFIG`, loaded from `config/config.yaml`) so paths, seeds,
model hyperparameter ranges, and fairness thresholds are defined in exactly
one place.

### 2.2 Module responsibilities

| Module | Responsibility |
|---|---|
| `dac.config` | Loads `config/config.yaml`, resolves all paths to absolute `Path` objects, creates directories on import. |
| `dac.data.synthetic` | Generates a schema-accurate synthetic UCI credit-card dataset, including a documented synthetic bias term. |
| `dac.data.loader` | Chooses real data (if prepared) vs. synthetic fallback; caches synthetic data as parquet. |
| `dac.features.engineering` | Derives interpretable features (utilization/repayment ratios, delinquency counts, AGE_GROUP bucketing); builds the shared `ColumnTransformer` preprocessing pipeline used by all four models. |
| `dac.eda.eda_report` | Produces missingness, target-balance, correlation, and protected-attribute breakdown figures + a markdown report. |
| `dac.models.train` | Builds each model family's `sklearn` `Pipeline` (preprocessor + estimator) and fits them; `compute_balanced_sample_weight` for estimators (EBM) with no built-in class-weighting. |
| `dac.models.tune` | `RandomizedSearchCV` for Logistic Regression; Optuna (TPE sampler) for XGBoost, CatBoost, and EBM, all optimizing CV ROC-AUC. |
| `dac.models.evaluate` | Threshold-free + threshold-based metrics (ROC-AUC, PR-AUC, F1, KS statistic, Brier score) and diagnostic plots (ROC, PR, calibration, confusion matrix). |
| `dac.explainability.shap_explain` | Global (bar, beeswarm) and local (waterfall) SHAP explanations for a fitted pipeline (LR/XGBoost/CatBoost). |
| `dac.explainability.ebm_explain` | EBM's own native global term-importance and shape-function plots (no post-hoc approximation needed); Spearman rank-agreement helper vs. another model's SHAP importances. |
| `dac.fairness.audit` | Fairlearn `MetricFrame`-based per-group audit: selection rate, demographic parity ratio/difference, equalized-odds difference, four-fifths-rule flag. |
| `dac.fairness.mitigation` | Kamiran & Calders (2012) reweighing: per-row sample weights that decorrelate a protected attribute from the label in the training set. |

### 2.3 Key design decisions

- **Config-driven, not hard-coded.** All paths, the random seed, split
  ratios, model hyperparameter defaults, tuning trial counts, and the
  fairness disparate-impact threshold live in `config/config.yaml`.
- **One shared preprocessing pipeline for all four models.** `build_preprocessor()`
  returns a single `ColumnTransformer` (median-impute + scale for numeric,
  most-frequent-impute + one-hot for categorical), reused identically by
  Logistic Regression, XGBoost, CatBoost, and EBM, so SHAP, EBM's native
  explanation, and the fairness audit are all comparable across models. It
  also uses `set_output(transform="pandas")` so EBM (which reads column
  names directly off its input) sees the same feature names SHAP is
  explicitly given.
- **Protected attributes are excluded from model features.** `SEX` and
  `AGE_GROUP` are never passed to any estimator — they are used only for the
  post-hoc fairness audit, matching standard fair-lending practice.
- **PR-AUC and Brier score alongside ROC-AUC.** The ~22% default rate makes
  ROC-AUC alone a reasonable but incomplete picture under class imbalance.
- **Reweighing over in-processing/post-processing mitigation.** Chosen
  because it is model-agnostic (identical code path for all four model
  families, since all four estimators accept `sample_weight`) and requires
  no architecture changes.
- **EBM gets its own explainer, not a SHAP wrapper.** EBM is ante-hoc
  interpretable — its own shape functions and term importances are exactly
  what the model used to make its predictions, so approximating it with
  SHAP would be a strictly worse substitute for the real thing (and the
  point of including EBM is to compare the two, not collapse them).
- **The four-fifths rule is a borrowed convention, not a legal threshold.**
  The 0.80 disparate-impact-ratio flag in `dac.fairness.audit` is an EEOC
  employment-discrimination heuristic (US), with no direct legal standing
  under ECOA/Reg B or outside the US. It's used here as a common
  fairness-ML convention for flagging practically significant gaps.
- **Real-data-if-present, synthetic-otherwise loading.** `dac.data.loader`
  never requires code changes to switch from synthetic to real data — it
  checks `data/raw/uci_credit/default_of_credit_card_clients.csv` first,
  falling back to a cached synthetic dataset only if that's absent.

### 2.4 Data flow

```
config/config.yaml --> dac.config.CONFIG --> every stage

data/external/UCI/*.xls --> scripts/prepare_uci_credit.py --> data/raw/uci_credit/
                                                                     |
                                                                     v
                                                      dac.data.loader.load_uci_credit()
                                                    (real file, else synthetic)
                                                                     |
                                                                     v
                                       dac.features.engineering (+ train/test split)
                                                                     |
                             +----------------------+----------------------+
                             v                      v                      v
                   dac.models.train      dac.models.tune          dac.eda.eda_report
                             |                      |                      |
                             +----------+-----------+                      v
                                        v                         reports/figures/*.png
                             dac.models.evaluate                  reports/model_performance/*.md
                                        |
                   +--------------------+--------------------+
                   v                    v                    v
      dac.explainability      dac.fairness.audit    dac.fairness.mitigation
      .shap_explain /          (pre-mitigation,       --> retrain --> dac.fairness.audit
      .ebm_explain             all 4 models)          (post-mitigation, all 4 models)
                   |                    |                    |
                   v                    v                    v
            reports/figures/    reports/model_performance/fairness/*.json
            .../shap/, .../ebm/  models/*.joblib
```

---

## 3. Project Structure

```
Code/
 +-- config/
 |    +-- config.yaml              # single source of truth: paths, seed, model/tuning/fairness params
 +-- data/
 |    +-- raw/uci_credit/          # prepared real CSV (gitignored)
 |    +-- processed/               # optional intermediate feature caches (gitignored)
 |    +-- external/UCI/            # raw source .xls export (gitignored)
 |    +-- README.md                # expected layout + how to prepare real data
 +-- docs/
 |    +-- DEVELOPER_GUIDE.md/.pdf  # this document
 +-- models/                       # trained model artifacts (*.joblib), gitignored
 +-- notebooks/                    # 01-05, one per pipeline stage group
 +-- reports/
 |    +-- figures/                 # all generated PNGs, gitignored
 |    +-- model_performance/       # metrics CSV/JSON/MD, gitignored
 +-- scripts/
 |    +-- prepare_uci_credit.py    # real-data preparation (xls -> csv, cleanup)
 |    +-- run_pipeline.py          # end-to-end orchestrator -- the main entry point
 |    +-- build_rq_flowchart.py    # RQ1-RQ4 lineage / solution-flow diagram
 |    +-- build_interim_report_uci.py  # Interim Report docx generator
 +-- src/dac/                      # the installable `dac` package
 |    +-- config.py
 |    +-- data/            (loader.py, synthetic.py)
 |    +-- features/        (engineering.py)
 |    +-- eda/             (eda_report.py)
 |    +-- models/          (train.py, tune.py, evaluate.py)
 |    +-- explainability/  (shap_explain.py, ebm_explain.py)
 |    +-- fairness/        (audit.py, mitigation.py)
 |    +-- utils/           (logging_utils.py)
 +-- tests/                        # pytest unit tests + one integration smoke test
 +-- requirements.txt
 +-- pyproject.toml                # packaging + pytest config
 +-- README.md
```

---

## 4. Setup Guide

### 4.1 Prerequisites

- Python 3.10+ (built and tested on 3.13)
- git
- ~2 GB free disk for the virtual environment + dependencies

### 4.2 Environment setup

```bash
cd Code
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .          # installs the `dac` package from src/ in editable mode
```

`requirements.txt` pins minimum versions for: `numpy`, `pandas`, `scikit-learn`,
`scipy`, `xgboost`, `catboost`, `interpret` (EBM), `optuna`, `shap`, `fairlearn`,
`matplotlib`, `seaborn`, `joblib`, `pyyaml`, `xlrd`, `pyarrow`, `pytest`,
`tabulate`, `jupyter`.

> **Windows note:** if `pip install` tries to compile `numpy` from source
> (slow, and can fail without a C/C++ toolchain), it usually means the
> resolver picked a version with no prebuilt wheel for your Python version.
> Upgrade pip first (`python -m pip install --upgrade pip`) and re-run.

### 4.3 Data setup

```bash
python scripts/prepare_uci_credit.py
```

Reads `data/external/UCI/default of credit card clients.xls`, applies the
documented cleanup (target-column rename, `SEX`/`EDUCATION`/`MARRIAGE`
label mapping), and writes `data/raw/uci_credit/default_of_credit_card_clients.csv`.
If this step is skipped, the pipeline transparently falls back to synthetic
data (see Section 1, Data availability) — no error, just a logged warning.

### 4.4 Verifying the install

```bash
pytest
```

Expect all unit tests plus one full-pipeline integration smoke test (runs
the entire pipeline, all four models, on a tiny synthetic sample) to pass.
This is the fastest way to confirm the environment is correctly set up
before a full run.

---

## 5. Running Guide

### 5.1 Full pipeline (recommended entry point)

```bash
python scripts/prepare_uci_credit.py   # once, or whenever the source .xls changes
python scripts/run_pipeline.py
```

Runs all nine stages described in Section 2.1 against the real (or
synthetic-fallback) UCI credit-card data, training and tuning all four model
families.

For a fast iteration/smoke-test loop (small synthetic sample, few tuning
trials):

```bash
python scripts/run_pipeline.py --quick
```

**Outputs:**

| Location | Contents |
|---|---|
| `reports/figures/uci_credit/` | EDA plots, evaluation suites, SHAP plots, EBM plots, fairness plots |
| `reports/model_performance/` | `eda_uci_credit.md`, `model_comparison.csv/json/png`, `fairness/*.json`, `run_summary.json` |
| `models/` | `<model>_tuned.joblib` for each of the 4 models, plus `<model>_tuned_mitigated.joblib` |

`reports/model_performance/run_summary.json` is the single consolidated
record of the whole run — shape, EDA stats, all four models' baseline/tuned
metrics, tuning best params, SHAP-vs-EBM agreement, and fairness results
before/after mitigation for all four models.

### 5.2 Notebooks (interactive / exploratory)

Open with `jupyter lab` or `jupyter notebook` from the `Code/` directory:

| Notebook | Covers |
|---|---|
| `01_eda_uci_credit.ipynb` | Data loading + EDA suite |
| `02_model_training_tuning.ipynb` | Baseline training, hyperparameter tuning, evaluation, all 4 models |
| `03_explainability_shap.ipynb` | SHAP global/local explanations |
| `04_fairness_audit_mitigation.ipynb` | Pre-mitigation audit, reweighing, post-mitigation re-audit |
| `05_ebm_explainability_comparison.ipynb` | EBM native explanation vs. SHAP; Spearman agreement (RQ2) |

### 5.3 Running individual stages from Python

```python
from dac.config import CONFIG
from dac.data.loader import load_uci_credit
from dac.eda.eda_report import run_eda

df, is_synthetic = load_uci_credit()
stats = run_eda(
    df,
    target_col="DEFAULT_PAYMENT_NEXT_MONTH",
    protected_attributes=["SEX", "AGE_GROUP"],
    figures_dir=CONFIG["paths"]["figures_dir"],
    report_path=CONFIG["paths"]["metrics_dir"] / "eda_uci_credit.md",
    dataset_name="uci_credit",
)
```

### 5.4 Tests

```bash
pytest                          # everything
pytest tests/test_models.py -v  # one module
pytest -k fairness               # by keyword
pytest tests/test_pipeline.py -v -s   # full-pipeline smoke test, streamed output
```

---

## 6. Interpreting the Outputs

- **`model_comparison.csv/json`** — ROC-AUC, PR-AUC, F1, KS statistic, and
  Brier score for every baseline and tuned model, side by side. The model
  with the highest ROC-AUC is automatically selected as "best" for the
  downstream SHAP/fairness/mitigation stages.
- **`eda_uci_credit.md`** — shape, duplicates, target balance, correlation
  with target, and the target rate by `SEX`/`AGE_GROUP` (the EDA-level early
  warning for the fairness audit that follows later in the pipeline).
- **`fairness_<model>_<attribute>.json`** — per-group selection rate,
  accuracy, TPR/FPR/FNR; demographic parity difference/ratio; equalized
  odds difference; and a boolean `disparate_impact_flag`. Compare
  `_pre_mitigation` and `_post_mitigation` files for the same model +
  attribute to see the effect of reweighing.
- **`run_summary.json`** — everything above in one file, plus SHAP/EBM top
  features, the SHAP-vs-EBM Spearman agreement, and fairness results for
  all four models, for a single-glance run record.
- **SHAP bar / beeswarm plots** — global feature importance and
  direction-of-effect for XGBoost (and the best model, if different);
  delinquency-status aggregates (`MAX_DELINQUENCY`, `MONTHS_DELINQUENT`) and
  credit-utilization ratios are the expected top drivers.
- **EBM shape-function plots** (`reports/figures/uci_credit/ebm/`) — the
  model's own learned per-feature contribution curves; compare visually and
  via the Spearman agreement figure against the SHAP importances for the
  same features.

---

## 7. Extending the Project

- **Add a model:** implement a `build_<name>_pipeline()` in
  `dac.models.train` returning an `sklearn.Pipeline` with steps
  `("preprocess", preprocessor)` and `("clf", estimator)`, add a matching
  `tune_<name>()` in `dac.models.tune`, and wire it into
  `scripts/run_pipeline.py`. Every downstream stage (SHAP, fairness audit)
  works unmodified as long as the pipeline follows this `preprocess` + `clf`
  step-naming convention and the estimator accepts `sample_weight` in `fit`.
- **Add a mitigation technique:** add a function to
  `dac.fairness.mitigation` (e.g. an `sklearn`-compatible adversarial
  debiasing or equalized-odds postprocessing wrapper) and call it from
  Section 9 of `run_pipeline.py` alongside `compute_reweighing_weights`.
- **Mitigate on more than one attribute:** `run_pipeline.py` currently
  mitigates only on `SEX` (`config/config.yaml`'s `fairness.mitigation_attribute`)
  — this is a scope decision, not a limitation of `dac.fairness.mitigation`.
  To extend, compute a joint reweighing over the Cartesian product of
  `SEX` x `AGE_GROUP` groups.
- **Change any tunable parameter:** edit `config/config.yaml`; every stage
  reads from `dac.config.CONFIG` at call time, so no source changes are
  needed for path, seed, split-ratio, tuning-trial-count, or
  fairness-threshold changes.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `pip install` tries to build `numpy`/`pandas` from source and fails | No prebuilt wheel matched for your Python version. Upgrade `pip` first, then reinstall. |
| `ImportError: Import xlrd failed` when running `prepare_uci_credit.py` | The source file is a legacy `.xls` (not `.xlsx`); install `xlrd>=2.0.1`. |
| Pipeline runs but shows `uci_credit_is_synthetic: true` | `scripts/prepare_uci_credit.py` hasn't been run yet (or its output CSV was deleted); run it to switch to real data. |
| `run_pipeline.py` runs but fairness plots look empty/flat | Check `reports/model_performance/fairness/*.json` for the actual per-group numbers — small synthetic samples (e.g. `--quick` mode) can have too few rows per group for a stable signal. |
| SHAP step is slow | It runs on `max_background=200` / `max_explain=500` rows by default (see `dac.explainability.shap_explain.explain_model`); reduce these for faster, less precise runs. |
| EBM tuning is slow | EBM fits are slower per-trial than XGBoost/CatBoost at this row count; reduce `tuning.ebm_n_trials` in `config/config.yaml`. |
| Tests fail after editing `config/config.yaml` | `tests/test_pipeline.py` monkeypatches `CONFIG` at test time — a real edit to the YAML file changes defaults for *all* runs including tests. Confirm the new values are still valid types/paths. |
| `ModuleNotFoundError: No module named 'dac'` | Run `pip install -e .` from `Code/` (Section 4.2) — the package must be installed in editable mode for both scripts and tests to import it. |

---

## 9. Appendix: Configuration Reference

`config/config.yaml` keys:

| Key | Meaning |
|---|---|
| `seed` | Global random seed used everywhere (data generation, splits, model init, tuning samplers). |
| `paths.*` | Relative paths (resolved to absolute at load time) for raw/processed data, models, reports, figures, metrics. Directories are auto-created on import. |
| `data.uci_credit.n_synthetic_rows` | Row count for the synthetic fallback generator. |
| `data.uci_credit.target_col` / `id_col` / `protected_attributes` | Column names used throughout the pipeline. |
| `split.test_size` / `val_size` | Train/test split fractions (`val_size` is currently reserved; the pipeline uses a single train/test split plus CV within training). |
| `models.logistic_regression.max_iter` | Default LR solver iteration cap. |
| `models.xgboost.n_estimators` / `tree_method` | Default (untuned) XGBoost baseline params. |
| `models.catboost.n_estimators` | Default (untuned) CatBoost baseline params. |
| `models.ebm.max_bins` | Default (untuned) EBM baseline param. |
| `tuning.n_trials` | Optuna trial count for XGBoost tuning. |
| `tuning.catboost_n_trials` / `ebm_n_trials` | Optuna trial counts for CatBoost / EBM tuning. |
| `tuning.cv_folds` | Stratified K-fold count used by LR/XGBoost/CatBoost tuning. |
| `tuning.scoring` | CV scoring metric for tuning (`roc_auc`). |
| `fairness.favorable_label` | Which class of the target counts as "favorable" for selection-rate purposes (`0` = repaid). |
| `fairness.disparate_impact_threshold` | The four-fifths-rule cutoff (default `0.80`). |
| `fairness.mitigation_method` | Currently informational; the implemented method is reweighing (`dac.fairness.mitigation`). |
| `fairness.mitigation_attribute` | Which protected attribute reweighing is computed on (default `SEX`). |
