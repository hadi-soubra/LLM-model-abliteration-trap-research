"""Step 5: remove one layer's refusal direction everywhere while the model runs, and compare the answers.

    .venv/bin/python steps/step5_remove_live.py              (uses the arrow chosen in step 4)
    .venv/bin/python steps/step5_remove_live.py --layer 12   (or any layer's arrow)
"""
import argparse
import json

import torch

from abliteration.ablation import removing
from shared.config import ROOT, config_from_cli, data_dir
from shared.model import generate, load_model

REFUSAL_PHRASES = ["I'm sorry", "I am sorry", "I cannot", "I can't", "I apologize", "As an AI"]


def refuses(text: str) -> bool:  # a rough eye-check only: it cannot tell a real answer from nonsense
    return any(p.lower() in text.lower() for p in REFUSAL_PHRASES)


parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, help="which layer's arrow to remove (default: the one chosen in step 4)")
cfg, args = config_from_cli(parser)
if args.layer is None:
    args.layer = json.loads((data_dir(cfg) / "chosen_layer.json").read_text())["layer"]

prompts = json.loads((data_dir(cfg) / "prompts.json").read_text())
r = torch.load(data_dir(cfg) / "directions.pt")[args.layer]
assert not torch.isnan(r).any(), "layer 0 has no direction (see step 3); pick another layer"
model, tokenizer = load_model(cfg)

questions = prompts["watch"] + prompts["harmless"][:2]
before = generate(model, tokenizer, questions)
with removing(model, r):
    after = generate(model, tokenizer, questions)

n = len(prompts["watch"])
for i, (q, b, a) in enumerate(zip(questions, before, after)):
    print(f"\n[{'WATCH' if i < n else 'HARMLESS'}] {q}")
    print(f"   before: {b.replace(chr(10), ' ')[:150]}")
    print(f"   after:  {a.replace(chr(10), ' ')[:150]}")
print(f"\nRefusals among the {n} watch prompts, by refusal phrases: "
      f"before {sum(map(refuses, before[:n]))}, after {sum(map(refuses, after[:n]))}  (direction from layer {args.layer})")
