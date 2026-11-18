"""Step 2: record what the model computes inside, for the harmful and the harmless prompts.

    .venv/bin/python steps/step2_record_activations.py
"""
import json

import torch

from abliteration.activations import record_last_word
from shared.config import ROOT, config_from_cli, data_dir
from shared.model import load_model, wrap

cfg, _ = config_from_cli()
prompts = json.loads((data_dir(cfg) / "prompts.json").read_text())
model, tokenizer = load_model(cfg)

print("What the model actually reads for the first harmful prompt:")
print("  ", repr(wrap(tokenizer, prompts["harmful"][0])))

acts = {}
for pile in ["harmful", "harmless"]:
    acts[pile] = record_last_word(model, tokenizer, prompts[pile], cfg["batch_size"])
    print(f"\n{pile}: shape {list(acts[pile].shape)}  (prompts, layers, numbers per layer)")

print("\nThe first 6 of the numbers for the first harmful prompt, entering layer 12:")
print("  ", [round(x, 2) for x in acts["harmful"][0, 12, :6].tolist()])

print("\nAverage size (length) of that list of numbers, entering each layer:")
sizes = acts["harmful"].norm(dim=-1).mean(dim=0).tolist()
print("  ", "  ".join(f"L{i}:{s:.0f}" for i, s in enumerate(sizes)))

torch.save(acts, data_dir(cfg) / "activations.pt")
print(f"\nSaved to {data_dir(cfg).relative_to(ROOT) / 'activations.pt'}")
