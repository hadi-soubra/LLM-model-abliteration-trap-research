"""Step 4: score every candidate arrow on the check prompts, and choose the best one.

    .venv/bin/python steps/step4_choose_layer.py
"""
import json

import torch

from abliteration.ablation import removing
from abliteration.scoring import adding, first_word_logits, kl_divergence, refusal_score, refusal_token_ids
from shared.config import ROOT, config_from_cli, data_dir
from shared.model import load_model

cfg, _ = config_from_cli()
prompts = json.loads((data_dir(cfg) / "prompts.json").read_text())
acts, arrows = torch.load(data_dir(cfg) / "activations.pt"), torch.load(data_dir(cfg) / "directions.pt")
model, tokenizer = load_model(cfg)
ids, harmful, harmless = refusal_token_ids(tokenizer), prompts["harmful_check"], prompts["harmless_check"]
first = lambda qs: first_word_logits(model, tokenizer, qs, cfg["batch_size"])  # noqa: E731

normal_harmless = first(harmless)
print(f"Normal model, refusal score: {refusal_score(first(harmful), ids).mean():.2f} on harmful check prompts, "
      f"{refusal_score(normal_harmless, ids).mean():.2f} on harmless ones\n")

rows = []
for L in range(1, len(arrows)):  # layer 0 has no arrow
    r = arrows[L]
    gap = (acts["harmful"][:, L] @ r).mean() - (acts["harmless"][:, L] @ r).mean()  # distance between the piles
    with removing(model, r):
        removal = refusal_score(first(harmful), ids).mean().item()
        side = kl_divergence(normal_harmless, first(harmless)).mean().item()
    with adding(model, L, gap * r):  # push harmless prompts toward the harmful side at this one layer
        addition = refusal_score(first(harmless), ids).mean().item()
    rows.append((L, removal, addition, side))

# Arditi's rule: skip the last 20% of layers; adding must cause refusal; removing must barely change harmless prompts
last_layer, limit = int(0.8 * len(arrows)) - 1, cfg["choose"]["max_side_effect"]
passing = lambda lim: [x for x in rows if x[0] <= last_layer and x[2] > 0 and x[3] < lim]  # noqa: E731
while not passing(limit) and limit < 1.0:
    limit *= 1.5  # small models are fragile: loosen the side-effect limit step by step if nothing passes
best = min(passing(limit), key=lambda x: x[1]) if passing(limit) else None

print("layer   removal (lower is better)   addition (must be > 0)   side effect (lower is better)")
for L, removal, addition, side in rows:
    mark = "  <- chosen" if best and L == best[0] else ("" if (L, removal, addition, side) in passing(limit) else "  (fails)")
    print(f"{L:5d}   {removal:25.2f}   {addition:22.2f}   {side:29.3f}{mark}")
print(f"\nRules: layers 1-{last_layer}, addition > 0, side effect < {limit:.3f}")
if best:
    (data_dir(cfg) / "chosen_layer.json").write_text(json.dumps({"layer": best[0], "side_effect_limit": limit}))
    print(f"Chosen: the arrow from layer {best[0]}. Saved to {data_dir(cfg).relative_to(ROOT) / 'chosen_layer.json'}")
else:
    print("No arrow passes the rules.")
