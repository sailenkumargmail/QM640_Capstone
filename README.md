# Credit Default Prediction: Explainable AI & Bias Mitigation

Capstone project (QM640) implementing the framework described in
`Synopsis_Credit_Default_XAI_Bias_Mitigation.docx`:
a credit-default scoring pipeline that is (1) accurate, (2) explainable, and
(3) fairness-audited and bias-mitigated.

**See [docs/DEVELOPER_GUIDE.pdf](docs/DEVELOPER_GUIDE.pdf) (or the
[Markdown source](docs/DEVELOPER_GUIDE.md)) for the full high-level design,
setup, and running guide.**

**Scope (MVP, per mentor/reviewer critique):**
- **Home Credit Default Risk** (Kaggle) — full pipeline: EDA, feature
  engineering, baseline + tuned models (Logistic Regression, XGBoost),
  evaluation, SHAP explainability, fairness audit, and bias mitigation
  (reweighing) on gender/age.
- **HMDA** (CFPB mortgage data) — used **only** to fairness-audit a model on
  race/sex. The real extract used here (2008-2017 legacy LAR, see below)
  contains **originated loans only** (no denials), so the audited target is a
  pricing proxy (`high_cost_flag`: was the loan higher-priced under the
  pre-2018 Reg Z rate-spread-reporting rule?), not approval/denial. It is
  never treated as a "default" model or compared apples-to-apples with Home
  Credit's target (see the mentor/reviewer critique's Cross-Question 1).

## Data

The pipeline runs against **real data** by default:

- **Home Credit Default Risk** (Kaggle): the full `application_train.csv`
  (307,511 rows) from `data/external/HCDR/home-credit-default-risk.zip`,
  extracted to `data/raw/home_credit/`.
- **HMDA**: `scripts/prepare_hmda_real.py` builds a nationwide sample from the
  legacy (pre-2018) LAR extracts in `data/external/HMDA/` (2007-2017, one zip
  per year). Two years (2007, 2011) shipped as truncated/corrupt zip files and
  are skipped automatically; the remaining 9 years (2008-2010, 2012-2017) are
  streamed in chunks and randomly subsampled at ~1% per year (reproducible,
  seeded per year) — full nationwide volume is ~70-90M rows across the decade,
  too large to load in memory on a typical dev machine and unnecessary for
  stable model/fairness metrics. The result (~716K rows) is written to
  `data/raw/hmda/hmda_nationwide_2008_2017_sample.csv`. Because this extract
  is originated-loans-only, the target is derived rather than a raw field:
  `high_cost_flag = rate_spread.notna()` (rate_spread was only reported,
  pre-2018, when a loan's APR exceeded the HOEPA/Reg Z threshold), and
  `rate_spread`/`hoepa_status` are dropped from the model's features to avoid
  leaking the label.

If `data/raw/home_credit/application_train.csv` or `data/raw/hmda/hmda_*.csv`
are absent, `dac.data.loader` transparently falls back to a schema-accurate
**synthetic dataset** (see `src/dac/data/synthetic.py`) with a small, clearly
documented synthetic bias term injected on the protected attributes, so the
fairness-audit/mitigation stages always have a real, known effect to detect
and correct — no code changes needed either way.

To (re)build real data from scratch:
```bash
# Home Credit (requires Kaggle account + accepted competition rules;
# credentials at ~/.kaggle/kaggle.json) -- OR extract data/external/HCDR/*.zip manually
python scripts/download_home_credit.py

# HMDA modern data via the public CFPB API (no auth required)
python scripts/download_hmda.py --year 2023 --states MI OH IN IL WI

# OR, HMDA legacy nationwide sample from data/external/HMDA/*.zip
python scripts/prepare_hmda_real.py --sample-frac 0.01
```

See `data/README.md` for the expected file layout.

## Project structure

```
Code/
├── config/config.yaml          # all tunable parameters (paths, seeds, model/tuning settings)
├── data/{raw,processed,external}
├── models/                     # trained model artifacts (joblib), gitignored
├── notebooks/                  # exploratory notebooks mirroring the src/ pipeline
├── reports/{figures,model_performance}  # generated EDA/eval/fairness output
├── scripts/
│   ├── download_home_credit.py
│   ├── download_hmda.py
│   └── run_pipeline.py         # end-to-end orchestrator (this is what to run)
├── src/dac/
│   ├── config.py
│   ├── data/                   # loading + synthetic data generation
│   ├── features/                # feature engineering + preprocessing
│   ├── eda/                     # EDA report generation
│   ├── models/                  # train / tune / evaluate
│   ├── explainability/          # SHAP
│   └── fairness/                # audit (Fairlearn) + mitigation (reweighing) + HMDA audit
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
python scripts/run_pipeline.py            # full run
python scripts/run_pipeline.py --quick    # fast smoke-test run (fewer tuning trials)
```

This runs, in order: EDA → feature engineering → baseline training → hyperparameter
tuning → evaluation → SHAP explainability → fairness audit → bias mitigation
(reweighing) + re-audit → HMDA fairness-only audit. All figures land in
`reports/figures/`, all metrics/tables in `reports/model_performance/`
(including a consolidated `run_summary.json`), and trained models in `models/`.

## Tests

```bash
pytest
```

`tests/test_pipeline.py` runs the entire pipeline end-to-end on a tiny
synthetic sample as an integration smoke test; the rest are unit tests per
module.

## Methodology notes / known limitations

- **Four-fifths rule**: the 0.80 disparate-impact-ratio threshold used in
  `dac.fairness.audit` is an EEOC employment-discrimination heuristic, not a
  codified lending-fairness standard under ECOA/Reg B. It's used here as a
  common fairness-ML convention, not a claimed regulatory threshold.
- **Reweighing** (Kamiran & Calders, 2012) was chosen as the single bias-
  mitigation technique for the MVP scope (vs. the original synopsis's
  reweighing + adversarial debiasing + equalized-odds postprocessing matrix)
  because it is model-agnostic and requires no architecture changes.
- **PR-AUC and Brier score** are reported alongside ROC-AUC/F1 because Home
  Credit's low (~8-10%) default rate makes ROC-AUC alone potentially
  misleading under class imbalance.
- Protected attributes (`CODE_GENDER`/`AGE_GROUP` for Home Credit;
  `derived_race`/`derived_sex` for HMDA) are **excluded from model features**
  and used only for post-hoc fairness auditing, matching standard fair-
  lending practice.
