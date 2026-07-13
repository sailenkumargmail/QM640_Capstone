# Data directory

This directory is gitignored (except this file and `.gitkeep` placeholders) —
data is downloaded/generated locally, not committed.

## Expected layout

```
data/
├── raw/
│   ├── home_credit/
│   │   ├── application_train.csv     # from scripts/download_home_credit.py
│   │   └── ...
│   ├── hmda/
│   │   ├── hmda_MI_2023.csv          # from scripts/download_hmda.py
│   │   └── ...
│   ├── home_credit_synthetic.parquet  # auto-generated fallback (cached)
│   └── hmda_synthetic.parquet         # auto-generated fallback (cached)
├── processed/                          # intermediate engineered features (optional cache)
└── external/                           # any supplementary reference data
```

## How data gets here

`dac.data.loader.load_home_credit()` and `load_hmda()` check for real files
first (`data/raw/home_credit/application_train.csv`,
`data/raw/hmda/hmda_*.csv`). If absent, they transparently fall back to a
schema-accurate synthetic dataset (`dac.data.synthetic`), generate it once,
and cache it as a parquet file in `data/raw/` so repeated pipeline runs don't
regenerate it.

## Getting real data

- **Home Credit Default Risk** (Kaggle competition, requires a Kaggle account
  that has accepted the competition rules + API credentials at
  `~/.kaggle/kaggle.json`): `python scripts/download_home_credit.py`
- **HMDA** (CFPB Data Browser API, public, no auth):
  `python scripts/download_hmda.py --year 2023 --states MI OH IN IL WI`

Kaggle competition data carries redistribution restrictions — do not commit
raw Home Credit CSVs to this (or any public) repository.
