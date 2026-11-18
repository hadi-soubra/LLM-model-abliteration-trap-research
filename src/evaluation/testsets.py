"""The ruler's test sets, read from the web into memory (nothing is stored). None of them is ever used to
build an attack or train a defense."""
import csv
import io
import json
import random

from shared.prompts import _fetch

HARMBENCH_URL = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def harmbench() -> list[dict]:
    """The 200 'standard' HarmBench behaviours (the others need an extra passage or ask for copyrighted text)."""
    rows = csv.DictReader(io.StringIO(_fetch(HARMBENCH_URL, "HarmBench")))
    return [{"id": r["BehaviorID"], "text": r["Behavior"].strip(), "category": r["SemanticCategory"]}
            for r in rows if r["FunctionalCategory"] == "standard"]


def xstest_safe() -> list[dict]:
    """The 250 safe XSTest questions: they only sound alarming, so refusing them is over-refusal."""
    rows = csv.DictReader(io.StringIO(_fetch(XSTEST_URL, "XSTest")))
    return [{"id": f"xstest/{r['id']}", "text": r["prompt"].strip(), "type": r["type"]} for r in rows if r["label"] == "safe"]


def gsm8k_test() -> list[dict]:
    """The 1,319 GSM8K test problems; the correct answer is the number after '####' in the worked solution."""
    lines = _fetch(GSM8K_URL, "GSM8K").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return [{"id": f"gsm8k/{i}", "text": r["question"].strip(), "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
            for i, r in enumerate(rows)]


def load_testsets(cfg: dict, prompts: dict) -> dict[str, list[dict]]:
    """All four sets, cut to the sizes in the config (a fixed shuffle first, so a cut is still a fair sample)."""
    n, rng = cfg["eval"], random.Random(cfg["seed"])

    def cut(rows, k):
        rows = list(rows)
        rng.shuffle(rows)
        return rows[:k]

    harmless = [{"id": f"alpaca/eval/{i}", "text": t} for i, t in enumerate(prompts["harmless_eval"])]
    return {"harmful": cut(harmbench(), n["n_harmful"]), "harmless": cut(harmless, n["n_harmless"]),
            "xstest": cut(xstest_safe(), n["n_xstest"]), "gsm8k": cut(gsm8k_test(), n["n_gsm8k"])}
