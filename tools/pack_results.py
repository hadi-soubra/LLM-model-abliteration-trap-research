"""Build the zips to send back from any run folder, for example a run made before the full zip existed.

    .venv\\Scripts\\python tools\\pack_results.py --config configs\\gpu_qwen35_9b.yaml --run night1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shared.config import ROOT, config_from_cli, data_dir  # noqa: E402
from shared.runs import pack_full, pack_summary, write_environment  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="night1", help="the run folder name inside results/<config>/")
cfg, args = config_from_cli(parser)
out = ROOT / "results" / cfg["name"] / args.run
if not out.exists():
    sys.exit(f"no run folder at {out}")
if not (out / "environment.txt").exists():
    write_environment(out, args.config)
tag = f"SEND_TO_HADI_{cfg['name']}_{args.run}"
pack_summary(out, data_dir(cfg), ROOT / f"{tag}_summary.zip")
pack_full(out, data_dir(cfg), ROOT / f"{tag}_full.zip")
print(f"Send Hadi PRIVATELY: {ROOT / (tag + '_full.zip')}\n(small, numbers only: {ROOT / (tag + '_summary.zip')})")
