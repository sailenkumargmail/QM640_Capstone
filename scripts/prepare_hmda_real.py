"""Prepare a real, nationwide HMDA dataset from the pre-2018 legacy LAR
extracts in data/external/HMDA/ (one zip per year, 2007-2017).

Why this exists (vs. scripts/download_hmda.py):
scripts/download_hmda.py pulls from the modern (2018+) CFPB Data Browser API,
whose schema (derived_race, derived_sex, action_taken 1-8 incl. denials) is
what dac/config.yaml and dac/fairness/hmda_audit.py originally assumed. The
files placed in data/external/HMDA/ are a different, legacy-era extract
(2007-2017, "nationwide_originated-records") with a different column schema
AND a different population: every row already has action_taken == 1 (loan
originated). There is no approval/denial variance to model in this extract.

Given that, the meaningful binary fairness question this data supports is
pricing, not access: for loans that WERE originated, was the loan
higher-priced? Pre-2018 Reg Z / HMDA rules required `rate_spread` to be
reported only when a loan's APR exceeded the reporting threshold, so
`rate_spread.notna()` is the standard proxy for a "higher-priced mortgage
loan" (HPML) flag in this era. That becomes the target: `high_cost_flag`.

This script:
  1. Skips 2007 (the provided zip is truncated / not a valid zip -- confirmed
     via `zipfile`; local header has a streamed-length data descriptor but no
     end-of-central-directory record).
  2. Streams each of the 2008-2017 CSVs (up to 5.6GB uncompressed each) in
     chunks directly out of its zip, at ~19M rows nationwide combined per
     year across the decade -- too large to load in full on a 16GB-RAM
     machine and unnecessary for stable model/fairness metrics. Takes a
     reproducible ~1% random subsample per chunk (seeded per year) instead,
     dropping columns that don't exist pre-2018 (property_value, LTV, DTI
     buckets, interest_rate) and dropping rate_spread/hoepa_status from the
     FEATURE set (kept only to derive the target -- including them as
     features would leak the label).
  3. Renames the survivors to the modern-schema-style names the rest of the
     codebase (dac.features.engineering.engineer_hmda_features,
     dac.data.synthetic) already uses: derived_race, derived_sex, income,
     loan_amount, etc.
  4. Writes one combined CSV to data/raw/hmda/, which dac.data.loader.load_hmda
     picks up automatically (globs data/raw/hmda/hmda_*.csv).

Usage:
    python scripts/prepare_hmda_real.py [--sample-frac 0.01]
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dac.config import CONFIG
from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)

EXTERNAL_DIR = CONFIG["paths"]["raw_dir"].parent / "external" / "HMDA"
OUT_DIR = CONFIG["paths"]["raw_dir"] / "hmda"

YEARS = list(range(2007, 2018))

RAW_USECOLS = [
    "as_of_year",
    "loan_type_name",
    "property_type_name",
    "loan_purpose_name",
    "owner_occupancy_name",
    "loan_amount_000s",
    "preapproval_name",
    "state_abbr",
    "applicant_race_name_1",
    "applicant_sex_name",
    "applicant_income_000s",
    "purchaser_type_name",
    "rate_spread",
    "hoepa_status",
    "lien_status_name",
    "agency_abbr",
    "population",
    "minority_population",
    "hud_median_family_income",
    "tract_to_msamd_income",
    "number_of_owner_occupied_units",
    "number_of_1_to_4_family_units",
]

RENAME_MAP = {
    "loan_type_name": "loan_type",
    "property_type_name": "property_type",
    "loan_purpose_name": "loan_purpose",
    "owner_occupancy_name": "occupancy_type",
    "preapproval_name": "preapproval",
    "state_abbr": "state_code",
    "applicant_income_000s": "income",
    "purchaser_type_name": "purchaser_type",
    "lien_status_name": "lien_status",
    "agency_abbr": "agency",
}

NOT_AVAILABLE_RACE = {
    "Information not provided by applicant in mail, Internet, or telephone application",
    "Not applicable",
}
NOT_AVAILABLE_SEX = {
    "Information not provided by applicant in mail, Internet, or telephone application",
    "Not applicable",
}


def _is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def _process_year(year: int, sample_frac: float, chunksize: int = 1_000_000) -> pd.DataFrame | None:
    zip_path = EXTERNAL_DIR / f"hmda_{year}_nationwide_originated-records_labels.zip"
    if not zip_path.exists():
        logger.warning("No file for year %d at %s -- skipping", year, zip_path)
        return None
    if not _is_valid_zip(zip_path):
        logger.warning("Year %d zip is corrupt/truncated -- skipping (%s)", year, zip_path)
        return None

    rng = np.random.default_rng(seed=year)
    t0 = time.time()
    n_total = 0
    parts: list[pd.DataFrame] = []

    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            for chunk in pd.read_csv(f, usecols=RAW_USECOLS, chunksize=chunksize, low_memory=False):
                n_total += len(chunk)
                if sample_frac < 1.0:
                    mask = rng.random(len(chunk)) < sample_frac
                    chunk = chunk.loc[mask]
                if len(chunk):
                    parts.append(chunk)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RAW_USECOLS)
    logger.info(
        "Year %d: scanned %d rows, sampled %d rows (%.1fs)",
        year, n_total, len(df), time.time() - t0,
    )
    return df


def _adapt_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME_MAP).copy()

    df["high_cost_flag"] = df["rate_spread"].notna().astype(int)
    df = df.drop(columns=["rate_spread", "hoepa_status"])

    df["derived_race"] = df["applicant_race_name_1"].where(
        ~df["applicant_race_name_1"].isin(NOT_AVAILABLE_RACE), "Race Not Available"
    )
    df["derived_race"] = df["derived_race"].fillna("Race Not Available")
    df = df.drop(columns=["applicant_race_name_1"])

    df["derived_sex"] = df["applicant_sex_name"].where(
        ~df["applicant_sex_name"].isin(NOT_AVAILABLE_SEX), "Sex Not Available"
    )
    df["derived_sex"] = df["derived_sex"].fillna("Sex Not Available")
    df = df.drop(columns=["applicant_sex_name"])

    df["loan_amount"] = df["loan_amount_000s"] * 1000
    df = df.drop(columns=["loan_amount_000s"])

    df.insert(0, "loan_id", [f"HMDA-{i:08d}" for i in range(1, len(df) + 1)])

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-frac", type=float, default=0.01, help="Per-row sample fraction, applied per year")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    yearly_frames = []
    for year in YEARS:
        df_year = _process_year(year, sample_frac=args.sample_frac)
        if df_year is not None and len(df_year):
            yearly_frames.append(df_year)

    if not yearly_frames:
        raise SystemExit("No usable HMDA year files found.")

    combined = pd.concat(yearly_frames, ignore_index=True)
    combined = _adapt_schema(combined)

    out_path = OUT_DIR / "hmda_nationwide_2008_2017_sample.csv"
    combined.to_csv(out_path, index=False)
    logger.info(
        "Wrote %d rows x %d cols to %s | high_cost_flag rate=%.4f",
        combined.shape[0], combined.shape[1], out_path, combined["high_cost_flag"].mean(),
    )


if __name__ == "__main__":
    main()
