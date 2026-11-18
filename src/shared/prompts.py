"""Download the harmful (AdvBench) and harmless (Alpaca) prompts, and pick the ones we use.

Both datasets are read straight from the web into memory. Nothing is stored on disk except the
picked prompts, which step 1 saves to data/prompts.json.
"""
import csv
import io
import json
import random
import urllib.request

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


def _fetch(url: str, name: str) -> str:
    """Download a file into memory, printing progress so a slow download doesn't look frozen."""
    with urllib.request.urlopen(url, timeout=120) as response:
        total = int(response.headers.get("Content-Length", 0)) / 1e6
        chunks, done = [], 0
        while chunk := response.read(1 << 16):
            chunks.append(chunk)
            done += len(chunk)
            print(f"\r  downloading {name}: {done / 1e6:.1f} of {total:.1f} MB", end="", flush=True)
        print()
    return b"".join(chunks).decode("utf-8")


def download_advbench() -> list[str]:
    """AdvBench: harmful requests. Each CSV row has a "goal" (the request) and a "target" (unused)."""
    goals = [row["goal"].strip() for row in csv.DictReader(io.StringIO(_fetch(ADVBENCH_URL, "AdvBench")))]
    return list(dict.fromkeys(goals))  # drop exact duplicates, keep the order


def download_alpaca() -> list[str]:
    """Alpaca: ordinary instructions. We keep only those that need no extra input text."""
    rows = json.loads(_fetch(ALPACA_URL, "Alpaca"))  # each row: instruction, input, output
    kept = [r["instruction"].strip() for r in rows if not r["input"].strip()]
    return list(dict.fromkeys(kept))


def pick_prompts(cfg: dict, harmful_all: list[str], harmless_all: list[str]) -> dict[str, list[str]]:
    """Shuffle each list with a fixed seed and take what we need, one group after another, so no prompt is in
    two groups. New groups are always taken after the old ones, so adding one never changes the others.
    """
    n = cfg["prompts"]
    rng = random.Random(cfg["seed"])
    harmful, harmless = list(harmful_all), list(harmless_all)
    rng.shuffle(harmful)
    rng.shuffle(harmless)
    a = n["n_harmful"] + n["n_watch"]  # where the harmful check prompts start
    return {
        "harmful": harmful[: n["n_harmful"]],
        "harmless": harmless[: n["n_harmless"]],
        "watch": harmful[n["n_harmful"] : a],
        "harmful_check": harmful[a : a + n["n_check"]],
        "harmless_check": harmless[n["n_harmless"] : n["n_harmless"] + n["n_check"]],
        "harmless_eval": harmless[n["n_harmless"] + n["n_check"] : n["n_harmless"] + n["n_check"] + n.get("n_harmless_eval", 0)],
    }
