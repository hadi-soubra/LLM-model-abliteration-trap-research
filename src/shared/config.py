"""Read a settings file (YAML) into a plain Python dictionary."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # the project folder (two levels above src/shared/)


def load_config(path: str) -> dict:
    """path can be relative to the project folder, e.g. "configs/laptop_qwen05b.yaml"."""
    full = Path(path) if Path(path).is_absolute() else ROOT / path
    with open(full) as f:
        return yaml.safe_load(f)
