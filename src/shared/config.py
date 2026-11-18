"""Read a settings file (YAML), and where each model's files go."""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # the project folder (two levels above src/shared/)


def load_config(path: str) -> dict:
    """path can be relative to the project folder, e.g. "configs/laptop_qwen05b.yaml"."""
    full = Path(path) if Path(path).is_absolute() else ROOT / path
    with open(full) as f:
        cfg = yaml.safe_load(f)
    cfg["name"] = full.stem  # e.g. "laptop_qwen05b": names this model's data folder
    return cfg


def config_from_cli(parser: argparse.ArgumentParser | None = None, default: str = "configs/laptop_qwen05b.yaml"):
    """Every step takes --config (default: the laptop model). Returns (config, the other arguments)."""
    parser = parser or argparse.ArgumentParser()
    parser.add_argument("--config", default=default, help="settings file, e.g. configs/gpu_qwen7b.yaml")
    args = parser.parse_args()
    return load_config(args.config), args


def data_dir(cfg: dict) -> Path:
    """data/<config name>/: each model keeps its own prompts, activations and directions."""
    path = ROOT / "data" / cfg["name"]
    path.mkdir(parents=True, exist_ok=True)
    return path
