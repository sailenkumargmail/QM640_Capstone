"""Data loading with automatic real-data-if-present, else-synthetic fallback.

If the real, prepared file exists at data/raw/uci_credit/ (produced by
scripts/prepare_uci_credit.py from the raw UCI .xls export), it is loaded.
Otherwise a schema-accurate synthetic dataset is generated so the pipeline
always has something to run against (see dac.data.synthetic).
"""
from __future__ import annotations

import pandas as pd

from dac.config import CONFIG
from dac.data.synthetic import generate_uci_credit_synthetic
from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_uci_credit(force_synthetic: bool = False) -> tuple[pd.DataFrame, bool]:
    """Returns (dataframe, is_synthetic)."""
    raw_dir = CONFIG["paths"]["raw_dir"] / "uci_credit"
    real_file = raw_dir / "default_of_credit_card_clients.csv"

    if not force_synthetic and real_file.exists():
        logger.info("Loading REAL UCI credit-card data from %s", real_file)
        return pd.read_csv(real_file), False

    logger.warning(
        "Real UCI credit-card data not found at %s -- using synthetic data. "
        "Run scripts/prepare_uci_credit.py to use real data.",
        real_file,
    )
    cache_path = CONFIG["paths"]["raw_dir"] / "uci_credit_synthetic.parquet"
    if cache_path.exists() and not force_synthetic:
        return pd.read_parquet(cache_path), True

    n_rows = CONFIG["data"]["uci_credit"]["n_synthetic_rows"]
    df = generate_uci_credit_synthetic(n_rows=n_rows, seed=CONFIG["seed"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df, True
