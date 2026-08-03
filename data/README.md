# Data directory

This directory is gitignored (except this file and `.gitkeep` placeholders) —
data is downloaded/prepared locally, not committed.

## Expected layout

```
data/
├── raw/
│   ├── uci_credit/
│   │   └── default_of_credit_card_clients.csv   # prepared by scripts/prepare_uci_credit.py
│   └── uci_credit_synthetic.parquet              # auto-generated fallback (cached)
├── processed/                                     # intermediate engineered features (optional cache)
└── external/
    └── UCI/default of credit card clients.xls     # raw UCI source export
```

## How data gets here

`dac.data.loader.load_uci_credit()` checks for the prepared real file first
(`data/raw/uci_credit/default_of_credit_card_clients.csv`). If absent, it
transparently falls back to a schema-accurate synthetic dataset
(`dac.data.synthetic`), generates it once, and caches it as a parquet file in
`data/raw/` so repeated pipeline runs don't regenerate it.

## Getting real data

The raw source file ships as `data/external/UCI/default of credit card
clients.xls` (Yeh, I-C., & Lien, C. (2009). The comparisons of data mining
techniques for the predictive accuracy of probability of default of credit
card clients. *Expert Systems with Applications*, *36*(2), 2473–2480; UCI
Machine Learning Repository, https://archive.ics.uci.edu/dataset/350).

To prepare it for the pipeline (renames the target column, maps `SEX` to
`Male`/`Female`, collapses undocumented `EDUCATION`/`MARRIAGE` codes into
`"Other/Unknown"`, writes a plain CSV):

```bash
python scripts/prepare_uci_credit.py
```

No further code changes are needed afterward — `dac.data.loader` detects the
prepared file under `data/raw/uci_credit/` and prefers it automatically over
the synthetic fallback.
