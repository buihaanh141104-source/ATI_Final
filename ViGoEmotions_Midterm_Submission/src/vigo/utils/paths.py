"""Project paths. Every output location is derived from the repository root."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESOURCES_DIR = PROJECT_ROOT / "resources"
RESULTS_DIR = Path(os.environ.get("VIGO_RESULTS_DIR", PROJECT_ROOT / "results"))
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
PRED_DIR = RESULTS_DIR / "predictions"
PRIVATE_DIR = RESULTS_DIR / "private"  # contains raw text -> never publish (gitignored)
SPLITS_DIR = PROJECT_ROOT / "splits"
CACHE_DIR = PROJECT_ROOT / ".cache"


def data_dir(override: str | os.PathLike | None = None) -> Path:
    """Resolve the directory holding train.csv / val.csv / test.csv / label_dict.json."""
    if override:
        return Path(override)
    if os.environ.get("VIGO_DATA_DIR"):
        return Path(os.environ["VIGO_DATA_DIR"])
    return PROJECT_ROOT / "data"


def ensure_dirs() -> None:
    for d in (RUNS_DIR, TABLES_DIR, FIGURES_DIR, PRED_DIR, PRIVATE_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
