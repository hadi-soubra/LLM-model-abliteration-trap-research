"""Check that this computer is ready for a night run, before starting one.

    .venv\\Scripts\\python tools\\preflight.py --config configs\\gpu_qwen7b.yaml      (GPU PC)
    .venv/bin/python tools/preflight.py                                          (laptop)

Prints OK or a plain-language problem for each check, then ALL GOOD or what to fix.
"""
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shared.config import config_from_cli  # noqa: E402

cfg, _ = config_from_cli()
problems = []


def check(name: str, ok: bool, detail: str, fix: str = "") -> None:
    print(f"{'OK     ' if ok else 'PROBLEM'}  {name}: {detail}")
    if not ok:
        problems.append(f"{name}: {fix or detail}")


check("Python", sys.version_info >= (3, 10), sys.version.split()[0], "Python 3.10 or newer is needed")
try:
    import torch
    gpu = torch.cuda.is_available()
    check("GPU visible to PyTorch", gpu, torch.cuda.get_device_name(0) if gpu else "no GPU found",
          "reinstall PyTorch with the cu128 command from the instructions")
    if gpu:
        gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        need = 24 if "7B" in cfg["model"] else 4
        check("GPU memory", gb >= need, f"{gb:.0f} GB (this config needs about {need} GB)")
        check("bf16 support", torch.cuda.is_bf16_supported(), "supported" if torch.cuda.is_bf16_supported() else "missing")
        x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        check("GPU calculation", torch.isfinite(x @ x).all().item(), f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
except ImportError:
    check("PyTorch", False, "not installed", "run the install commands from the instructions")

cache = Path.home() / ".cache" / "huggingface"
cache.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(cache).free / 2**30
check("Free disk space", free >= 40, f"{free:.0f} GB free where models are stored ({cache})", "free up at least 40 GB")

for name, url in [("AdvBench", "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"),
                  ("Alpaca", "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"),
                  ("HarmBench", "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"),
                  ("XSTest", "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"),
                  ("GSM8K", "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl")]:
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30)
        check(f"Download {name}", True, "reachable")
    except Exception as e:
        check(f"Download {name}", False, type(e).__name__, "check the internet connection")

from huggingface_hub import hf_hub_download, whoami  # noqa: E402
try:
    check("Hugging Face login", True, f"logged in as {whoami()['name']}")
except Exception:
    check("Hugging Face login", False, "not logged in", "run:  .venv\\Scripts\\hf auth login   and paste Hadi's token")
judges = [cfg["judge"]["name"]] + ([cfg["judge"]["fallback"]] if cfg["judge"].get("fallback") else [])
available = []
for model in [cfg["model"]] + judges:
    try:
        hf_hub_download(model, "config.json")
        available.append(model)
        check(f"Model access {model}", True, "OK")
    except Exception as e:
        is_judge = model in judges
        check(f"Model access {model}", is_judge and len(judges) > 1, type(e).__name__ + (" (a fallback judge exists)" if is_judge else ""),
              "ask Hadi: the model may need an access request on Hugging Face")
if not any(j in available for j in judges):
    problems.append("no judge model is available")

print("\nALL GOOD: ready for the night run." if not problems else "\nFIX THESE FIRST:\n  - " + "\n  - ".join(problems))
sys.exit(1 if problems else 0)
