"""Load and resolve the project's YAML configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve all "paths" entries to absolute Path objects rooted at PROJECT_ROOT.
    for key, rel_path in cfg["paths"].items():
        abs_path = (PROJECT_ROOT / rel_path).resolve()
        abs_path.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = abs_path

    return cfg


CONFIG = load_config()
