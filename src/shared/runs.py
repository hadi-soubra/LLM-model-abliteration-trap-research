"""Packing a run's results to send back from the GPU PC: a full zip with everything, and a small numbers-only zip."""
import datetime as dt
import importlib.metadata
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path

from .config import ROOT


def _command(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=60).stdout.strip()
    except Exception as e:  # e.g. git or nvidia-smi not found
        return f"(not available: {type(e).__name__})"


def write_environment(out: Path, config_path: str) -> None:
    """What ran, and where: the code version, the config, the GPU and driver, every installed library version."""
    import torch

    lines = [f"written: {dt.datetime.now():%Y-%m-%d %H:%M}", f"code version (git commit): {_command('git', 'rev-parse', 'HEAD')}",
             f"uncommitted changes: {_command('git', 'status', '--short') or 'none'}", f"computer: {platform.platform()}",
             f"python: {platform.python_version()}", f"torch: {torch.__version__} (CUDA {torch.version.cuda})",
             f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", "",
             "nvidia-smi:", _command("nvidia-smi"), "", "installed libraries:"]
    lines += sorted(f"  {d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (out / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy(ROOT / config_path if not Path(config_path).is_absolute() else config_path, out / "config.yaml")


def pack_full(out: Path, data: Path, zip_path: Path) -> None:
    """Everything: the run folder (results, logs, step outputs, raw answers) and the model's data folder
    (prompts, activations, directions, chosen layer). Contains harmful text: send it privately."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, prefix in ((out, "results"), (data, "data")):
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    z.write(f, f"{prefix}/{f.relative_to(folder).as_posix()}")


def pack_summary(out: Path, data: Path, zip_path: Path) -> None:
    """Numbers and logs only: no prompts, no answers. Small enough for any messaging app."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.glob("*")):
            if f.is_file() and f.suffix in (".json", ".md", ".txt", ".yaml") and not f.name.startswith(("step1", "step2")):
                z.write(f, f.name)
        for name in ("chosen_layer.json", "step4_scores.json", "directions.pt"):
            if (data / name).exists():
                z.write(data / name, name)
