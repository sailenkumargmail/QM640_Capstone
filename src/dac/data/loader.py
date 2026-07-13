"""Data loading with automatic real-data-if-present, else-synthetic fallback.

If real files exist under data/raw/home_credit/ (from
scripts/download_home_credit.py) or data/raw/hmda/ (from
scripts/download_hmda.py), those are loaded. Otherwise a schema-accurate
synthetic dataset is generated so the pipeline always has something to run
against (see dac.data.synthetic).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dac.config import CONFIG
from dac.data.synthetic import generate_hmda_synthetic, generate_home_credit_synthetic
from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_home_credit(force_synthetic: bool = False) -> tuple[pd.DataFrame, bool]:
    """Returns (dataframe, is_synthetic)."""
    raw_dir = CONFIG["paths"]["raw_dir"] / "home_credit"
    real_file = raw_dir / "application_train.csv"

    if not force_synthetic and real_file.exists():
        logger.info("Loading REAL Home Credit data from %s", real_file)
        return pd.read_csv(real_file), False

    logger.warning(
        "Real Home Credit data not found at %s -- using synthetic data. "
        "Run scripts/download_home_credit.py with Kaggle credentials to use real data.",
        real_file,
    )
    cache_path = CONFIG["paths"]["raw_dir"] / "home_credit_synthetic.parquet"
    if cache_path.exists() and not force_synthetic:
        return pd.read_parquet(cache_path), True

    n_rows = CONFIG["data"]["home_credit"]["n_synthetic_rows"]
    df = generate_home_credit_synthetic(n_rows=n_rows, seed=CONFIG["seed"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df, True


def load_hmda(force_synthetic: bool = False) -> tuple[pd.DataFrame, bool]:
    """Returns (dataframe, is_synthetic)."""
    raw_dir = CONFIG["paths"]["raw_dir"] / "hmda"
    real_files = sorted(raw_dir.glob("hmda_*.csv")) if raw_dir.exists() else []

    if not force_synthetic and real_files:
        logger.info("Loading REAL HMDA data from %d file(s) in %s", len(real_files), raw_dir)
        return pd.concat([pd.read_csv(f) for f in real_files], ignore_index=True), False

    logger.warning(
        "Real HMDA data not found in %s -- using synthetic data. "
        "Run scripts/download_hmda.py to use real data.",
        raw_dir,
    )
    cache_path = CONFIG["paths"]["raw_dir"] / "hmda_synthetic.parquet"
    if cache_path.exists() and not force_synthetic:
        return pd.read_parquet(cache_path), True

    n_rows = CONFIG["data"]["hmda"]["n_synthetic_rows"]
    df = generate_hmda_synthetic(n_rows=n_rows, seed=CONFIG["seed"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df, True
