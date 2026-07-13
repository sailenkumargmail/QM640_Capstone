"""Download the real Home Credit Default Risk dataset from Kaggle.

Requires a Kaggle account that has accepted the competition rules, and API
credentials at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY env
vars). Get credentials from https://www.kaggle.com/settings -> "Create New
Token".

Usage:
    python scripts/download_home_credit.py

Writes CSVs to data/raw/home_credit/. Once real files exist there,
dac.data.loader.load_home_credit() will use them automatically instead of
the synthetic generator.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dac.config import CONFIG
from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)

COMPETITION = "home-credit-default-risk"


def main() -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as exc:
        logger.error(
            "Kaggle credentials not found. Place kaggle.json in ~/.kaggle/ "
            "or set KAGGLE_USERNAME/KAGGLE_KEY env vars. Original error: %s", exc
        )
        raise SystemExit(1)

    out_dir = CONFIG["paths"]["raw_dir"] / "home_credit"
    out_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading competition files for '%s' to %s", COMPETITION, out_dir)
    api.competition_download_files(COMPETITION, path=str(out_dir), quiet=False)

    zip_path = out_dir / f"{COMPETITION}.zip"
    if zip_path.exists():
        logger.info("Extracting %s", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        zip_path.unlink()

    logger.info("Done. Files in %s: %s", out_dir, sorted(p.name for p in out_dir.glob("*.csv")))


if __name__ == "__main__":
    main()
