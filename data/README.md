# Data directory

This directory is gitignored (except this file and `.gitkeep` placeholders) —
data is downloaded/generated locally, not committed.

## Expected layout

```
data/
├── raw/
│   ├── home_credit/
│   │   ├── application_train.csv     # extracted from external/HCDR/*.zip (or scripts/download_home_credit.py)
│   │   └── HomeCredit_columns_description.csv
│   ├── hmda/
│   │   └── hmda_nationwide_2008_2017_sample.csv   # from scripts/prepare_hmda_real.py
│   ├── home_credit_synthetic.parquet  # auto-generated fallback (cached)
│   └── hmda_synthetic.parquet         # auto-generated fallback (cached)
├── processed/                          # intermediate engineered features (optional cache)
└── external/
    ├── HCDR/home-credit-default-risk.zip                       # raw Kaggle download
    └── HMDA/hmda_<year>_nationwide_originated-records_labels.zip # raw legacy LAR extracts, 2007-2017
```

## How data gets here

`dac.data.loader.load_home_credit()` and `load_hmda()` check for real files
first (`data/raw/home_credit/application_train.csv`,
`data/raw/hmda/hmda_*.csv`). If absent, they transparently fall back to a
schema-accurate synthetic dataset (`dac.data.synthetic`), generate it once,
and cache it as a parquet file in `data/raw/` so repeated pipeline runs don't
regenerate it.

## Getting real data

- **Home Credit Default Risk**: the full dataset ships as
  `data/external/HCDR/home-credit-default-risk.zip`; only `application_train.csv`
  (the only file with labels) is extracted to `data/raw/home_credit/`. To
  re-download instead (requires a Kaggle account that has accepted the
  competition rules + API credentials at `~/.kaggle/kaggle.json`):
  `python scripts/download_home_credit.py`
- **HMDA**: two sources are supported —
  - Legacy nationwide LAR extracts (2007-2017) ship as one zip per year under
    `data/external/HMDA/`. `python scripts/prepare_hmda_real.py` streams each
    (chunked, since files run up to 5.6GB uncompressed), skips any
    truncated/corrupt year, takes a reproducible ~1% per-year subsample, adapts
    legacy column names to the modern schema (`derived_race`, `derived_sex`,
    `income`, `loan_amount`), and derives `high_cost_flag` from `rate_spread`
    (see README.md § Data for why — this extract is originated-loans-only, so
    there's no approval/denial field to model). Writes
    `data/raw/hmda/hmda_nationwide_2008_2017_sample.csv`.
  - Modern per-state/year data via the public CFPB Data Browser API (no auth):
    `python scripts/download_hmda.py --year 2023 --states MI OH IN IL WI`

Kaggle competition data carries redistribution restrictions — do not commit
raw Home Credit CSVs to this (or any public) repository.
