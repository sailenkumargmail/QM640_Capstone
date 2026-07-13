"""Download real HMDA Loan/Application Register (LAR) data via the public
CFPB HMDA Platform API (no authentication required).

API docs: https://ffiec.cfpb.gov/documentation/api/data-browser/

Usage:
    python scripts/download_hmda.py --year 2023 --states MI OH IN

Writes one CSV per requested state to data/raw/hmda/. Once real files exist
there, dac.data.loader.load_hmda() will use them automatically instead of
the synthetic generator.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dac.config import CONFIG
from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)

BASE_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"


def download_state_year(state: str, year: int, out_dir: Path) -> None:
    params = {"states": state, "years": year}
    logger.info("Requesting HMDA LAR data: state=%s year=%s", state, year)
    resp = requests.get(BASE_URL, params=params, timeout=120)
    resp.raise_for_status()

    out_path = out_dir / f"hmda_{state}_{year}.csv"
    out_path.write_bytes(resp.content)
    logger.info("Saved %s (%.1f KB)", out_path, len(resp.content) / 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--states", nargs="+", default=["MI", "OH", "IN", "IL", "WI"])
    args = parser.parse_args()

    out_dir = CONFIG["paths"]["raw_dir"] / "hmda"
    out_dir.mkdir(parents=True, exist_ok=True)

    for state in args.states:
        download_state_year(state, args.year, out_dir)


if __name__ == "__main__":
    main()
