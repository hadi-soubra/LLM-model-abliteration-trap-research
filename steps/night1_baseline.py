"""Night 1 (Experiment 2): the baseline. The model with no attack, after A1, and after A1b (two versions),
each measured with the ruler (judge on HarmBench, false alarms on harmless requests, XSTest, GSM8K).

    GPU PC:          .venv\\Scripts\\python steps\\night1_baseline.py --config configs\\gpu_qwen7b.yaml
    laptop rehearsal: .venv/bin/python steps/night1_baseline.py --quick      (4 questions per test set)

If it stops halfway, run the same command again: finished parts are skipped.
At the end it writes SEND_TO_HADI_<config>_night1_full.zip (everything, including raw answers: send it privately)
and SEND_TO_HADI_<config>_night1_summary.zip (numbers only).
"""
import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
import traceback

import torch

from abliteration.direction import harmless_levels
from abliteration.weights import edit_weights
from evaluation.evaluate import answer_testsets, judge_rows, summarize
from evaluation.judge import load_judge
from evaluation.testsets import load_testsets
from shared.config import ROOT, config_from_cli, data_dir
from shared.model import load_model
from shared.runs import pack_full, pack_summary, write_environment

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles otherwise crash on some characters
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="4 questions per test set: checks that everything runs")
cfg, args = config_from_cli(parser)
if args.quick:
    cfg["eval"].update(n_harmful=4, n_harmless=4, n_xstest=4, n_gsm8k=4)
out = ROOT / "results" / cfg["name"] / ("night1_quick" if args.quick else "night1")
raw = out / "raw"  # raw answers include harmful text: they never leave this computer
raw.mkdir(parents=True, exist_ok=True)
tag = f"SEND_TO_HADI_{cfg['name']}_night1{'_quick' if args.quick else ''}"
full_zip, summary_zip = ROOT / f"{tag}_full.zip", ROOT / f"{tag}_summary.zip"
log_file = open(out / "log.txt", "a", encoding="utf-8")
seconds = {}


def log(message: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def make_zips() -> None:
    pack_summary(out, data_dir(cfg), summary_zip)
    pack_full(out, data_dir(cfg), full_zip)


def tell_what_to_send(headline: str) -> None:
    mb = full_zip.stat().st_size / 1e6
    log(f"{headline}\nPlease send Hadi this file PRIVATELY (Google Drive shared only with him, or WeTransfer), "
        f"because it contains the model's raw answers:\n    {full_zip}   ({mb:.0f} MB)\n"
        f"If it is too big to send, send this small one instead (numbers only):\n    {summary_zip}")


def run_step(step: str, result_file: str) -> None:
    """Steps 1-4 of Experiment 1, on this model. Their printed output is kept in results/."""
    path = data_dir(cfg) / result_file
    if path.exists() and (result_file != "prompts.json" or "harmless_eval" in json.loads(path.read_text())):
        log(f"{step}: already done, skipped")
        return
    log(f"{step}: running")
    t0 = time.time()
    done = subprocess.run([sys.executable, str(ROOT / "steps" / f"{step}.py"), "--config", args.config],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONUTF8": "1"})
    (out / f"{step}.txt").write_text(done.stdout + done.stderr, encoding="utf-8")
    seconds[step] = round(time.time() - t0)
    if done.returncode != 0 or not (data_dir(cfg) / result_file).exists():
        raise RuntimeError(f"{step} failed; its output is in {out / (step + '.txt')}")
    log(f"{step}: done in {seconds[step]} s")


def main() -> None:
    write_environment(out, args.config)
    log(f"Night 1 on {cfg['model']} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    for step, result in [("step1_get_prompts", "prompts.json"), ("step2_record_activations", "activations.pt"),
                         ("step3_find_direction", "directions.pt"), ("step4_choose_layer", "chosen_layer.json")]:
        run_step(step, result)

    d = data_dir(cfg)
    layer = json.loads((d / "chosen_layer.json").read_text())["layer"]
    r = torch.load(d / "directions.pt")[layer]
    per_layer = harmless_levels(torch.load(d / "activations.pt")["harmless"], r)
    conditions = {"none": None, "a1": [], "a1b_per_layer": per_layer, "a1b_constant": [per_layer[layer]] * len(per_layer)}
    (out / "levels.json").write_text(json.dumps({"layer": layer, "per_layer": per_layer}, indent=1))
    log(f"chosen arrow: layer {layer}; harmless level there {per_layer[layer]:.3f}")

    sets = load_testsets(cfg, json.loads((d / "prompts.json").read_text()))
    log("test sets: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
    for name, levels in conditions.items():  # answers, one model at a time
        if (raw / f"{name}.jsonl").exists():
            log(f"{name}: answers already done, skipped")
            continue
        log(f"{name}: loading the model and answering")
        t0 = time.time()
        model, tokenizer = load_model(cfg)
        if levels is not None:  # [] means A1 (dial set to 0), a list of levels means A1b
            edit_weights(model, r, levels or None)
        rows = answer_testsets(model, tokenizer, sets, cfg)
        (raw / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        seconds[f"answers_{name}"] = round(time.time() - t0)
        log(f"{name}: done in {seconds[f'answers_{name}']} s")
        del model
        torch.cuda.empty_cache()

    judge = None
    results = {}
    for name in conditions:  # the judge, loaded once for all conditions
        rows = [json.loads(x) for x in (raw / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        if not all("judge" in x for x in rows if x["set"] in ("harmful", "harmless")):
            if judge is None:
                log("loading the judge")
                judge = load_judge(cfg)
                log(f"judge: {judge.name}")
            t0 = time.time()
            judge_rows(rows, judge)
            (raw / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
            seconds[f"judge_{name}"] = round(time.time() - t0)
        results[name] = summarize(rows)
        log(f"{name}: judged")

    table = ["| condition | harmful answers (judge) | judge false alarms | GSM8K | XSTest refused | refusal phrases on harmful | cut off (harmful) |",
             "|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        table.append(f"| {name} | {m['harmful'].get('unsafe')} | {m['harmless'].get('unsafe')} | {m['gsm8k']['accuracy']} | "
                     f"{m['xstest']['refusal_phrases']} | {m['harmful']['refusal_phrases']} | {m['harmful']['cut_off']} |")
    (out / "table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    info = {"model": cfg["model"], "judge": judge.name if judge else "already judged", "chosen_layer": layer,
            "test_set_sizes": {k: len(v) for k, v in sets.items()}, "decoding": cfg["decoding"], "seconds": seconds,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "python": platform.python_version(), "finished": f"{dt.datetime.now():%Y-%m-%d %H:%M}"}
    (out / "metrics.json").write_text(json.dumps({"info": info, "results": results}, indent=1), encoding="utf-8")
    log("\n" + "\n".join(table))


try:
    main()
    make_zips()
    tell_what_to_send("FINISHED.")
except Exception:
    log("STOPPED because of an error:\n" + traceback.format_exc())
    make_zips()
    tell_what_to_send("The files below contain the log of what went wrong.")
    sys.exit(1)
