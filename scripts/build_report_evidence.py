"""Generates the data-preview and repo-tree screenshot images referenced in
the Interim Report's "GitHub Data Availability Statement / Screenshots"
section, directly from the checked-in data files and repo layout.

Usage:
    python scripts/build_report_evidence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dac.data.loader import load_uci_credit  # noqa: E402

OUT_DIR = REPO / "reports" / "figures" / "report_evidence"

EXCLUDE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "catboost_info", ".kaggle"}


def build_preview():
    df, _ = load_uci_credit()
    cols = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1", "DEFAULT_PAYMENT_NEXT_MONTH"]
    preview = df[cols].head(8).rename(columns={"DEFAULT_PAYMENT_NEXT_MONTH": "DEFAULT"})

    fig, ax = plt.subplots(figsize=(13, 3.1))
    ax.axis("off")
    tbl = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f2f2f2" if r % 2 == 0 else "white")
    fig.suptitle(
        "data/raw/uci_credit/default_of_credit_card_clients.csv -- first 8 rows, key columns",
        fontsize=10, fontweight="bold", y=0.98,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "preview_uci_credit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", OUT_DIR / "preview_uci_credit.png")


def build_repo_tree(max_depth: int = 2):
    lines = [f"{REPO.name}/"]

    def walk(path: Path, prefix: str, depth: int):
        if depth > max_depth:
            return
        entries = sorted(
            [
                p for p in path.iterdir()
                if p.name not in EXCLUDE_DIRS and not p.name.startswith(".") and "egg-info" not in p.name
            ],
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "+-- "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                walk(entry, prefix + ("    " if is_last else "|   "), depth + 1)

    walk(REPO, "", 1)
    text = "\n".join(lines)

    line_h = 0.185
    fig_h = line_h * (len(lines) + 2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(lines) + 2)
    ax.invert_yaxis()
    ax.axis("off")
    ax.text(0.01, 0.3, "Repository folder structure", fontsize=11, fontweight="bold", va="top")
    for i, line in enumerate(lines):
        ax.text(0.01, i + 1.3, line, family="monospace", fontsize=9, va="top", ha="left")
    fig.savefig(OUT_DIR / "repo_tree.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("Wrote", OUT_DIR / "repo_tree.png")


if __name__ == "__main__":
    build_preview()
    build_repo_tree()
