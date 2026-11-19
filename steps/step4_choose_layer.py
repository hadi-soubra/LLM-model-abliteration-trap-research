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
normal = {"harmful": refusal_score(first(harmful), ids).mean().item(), "harmless": refusal_score(normal_harmless, ids).mean().item()}
halfway = (normal["harmful"] + normal["harmless"]) / 2  # addition must lift harmless prompts at least this far
print(f"Normal model, refusal score: {normal['harmful']:.2f} on harmful check prompts, {normal['harmless']:.2f} on harmless ones\n")

rows = []
for L in range(1, len(arrows)):  # layer 0 has no arrow
    r = arrows[L]
    gap = (acts["harmful"][:, L] @ r).mean() - (acts["harmless"][:, L] @ r).mean()  # distance between the piles
    with removing(model, r):
        removal = refusal_score(first(harmful), ids).mean().item()
        side = kl_divergence(normal_harmless, first(harmless)).mean().item()
    with adding(model, L, gap * r):  # push harmless prompts toward the harmful side at this one layer
        addition = refusal_score(first(harmless), ids).mean().item()
    rows.append({"layer": L, "removal": removal, "addition": addition, "side_effect": side})

# rules: skip the last 20% of layers (Arditi et al.); adding must move harmless prompts at least halfway to the harmful
# ones; removing must barely change harmless prompts. Loosen the side-effect limit only if it is what blocks every arrow.
last_layer, limit = int(0.8 * len(arrows)) - 1, cfg["choose"]["max_side_effect"]
fails = lambda x, lim: [why for why, bad in (("layer", x["layer"] > last_layer), ("addition", x["addition"] <= halfway),  # noqa: E731
                                              ("side effect", x["side_effect"] >= lim)) if bad]
blocked_by_side_effect_only = lambda lim: any(fails(x, lim) == ["side effect"] for x in rows)  # noqa: E731
while not any(not fails(x, limit) for x in rows) and blocked_by_side_effect_only(limit) and limit < 1.0:
    limit *= 1.5
passing = [x for x in rows if not fails(x, limit)]
fallback = not passing  # unattended runs must not stop here: take the best allowed layer, and say so loudly
pool = passing or [x for x in rows if x["layer"] <= last_layer]
best = min(pool, key=lambda x: x["removal"])

print("layer   removal (lower is better)   addition (higher is better)   side effect (lower is better)")
for x in rows:
    mark = "  <- chosen" if x is best else ("" if x in passing else "  (fails: " + ", ".join(fails(x, limit)) + ")")
    print(f"{x['layer']:5d}   {x['removal']:25.2f}   {x['addition']:27.2f}   {x['side_effect']:29.3f}{mark}")
print(f"\nRules: layers 1-{last_layer}, addition > {halfway:.2f} (halfway from harmless to harmful), side effect < {limit:.3f}")
if fallback:
    print("WARNING: no arrow passes every rule; chose the allowed layer with the best removal anyway")
result = {"layer": best["layer"], "side_effect_limit": limit, "addition_threshold": halfway, "fallback": fallback}
(data_dir(cfg) / "chosen_layer.json").write_text(json.dumps(result))
(data_dir(cfg) / "step4_scores.json").write_text(json.dumps({"normal": normal, **result, "last_layer": last_layer, "rows": rows}, indent=1))
print(f"Chosen: the arrow from layer {best['layer']}. Saved to {data_dir(cfg).relative_to(ROOT) / 'chosen_layer.json'}")
