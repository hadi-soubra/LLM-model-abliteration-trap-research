"""Step 6: make the removal permanent in the weights, and check that it matches the hooks of step 5.

    .venv/bin/python steps/step6_edit_weights.py
"""
import json
import os

import torch

from abliteration.ablation import removing
from abliteration.activations import record_last_word
from abliteration.scoring import first_word_logits, kl_divergence
from abliteration.weights import edit_weights
from shared.config import ROOT, load_config
from shared.model import generate, load_model

cfg = load_config("configs/laptop_qwen05b.yaml")
prompts = json.loads((ROOT / "data" / "prompts.json").read_text())
layer = json.loads((ROOT / "data" / "chosen_layer.json").read_text())["layer"]
r = torch.load(ROOT / "data" / "directions.pt")[layer]
model, tokenizer = load_model(cfg)
questions, bs = prompts["watch"] + prompts["harmless"][:2], cfg["batch_size"]


def dial_inside():  # largest dial reading (along r) entering any layer, at the last word of the watch prompts
    return (record_last_word(model, tokenizer, prompts["watch"], bs) @ r).abs().max().item()


print(f"Arrow from layer {layer}. Largest dial reading inside the normal model: {dial_inside():.2f}")
with removing(model, r):  # step 5's way: hooks
    hooked, hooked_logits = generate(model, tokenizer, questions), first_word_logits(model, tokenizer, questions, bs)

edit_weights(model, r)  # step 6's way: change the weights; no hooks from here on
edited, edited_logits = generate(model, tokenizer, questions), first_word_logits(model, tokenizer, questions, bs)
print(f"Largest dial reading inside the edited model: {dial_inside():.3f}  (should be about 0)")
print(f"First-word probabilities, hooks vs edited weights: largest difference (KL) "
      f"{kl_divergence(hooked_logits, edited_logits).max():.5f}  (0 = identical)\n")

for q, h, e in zip(questions, hooked, edited):
    if h == e:
        print(f"same answer     {q[:70]}")
    else:
        k = len(os.path.commonprefix([h, e]))
        print(f"differs at character {k:3d} of {len(h)}: {q[:70]}\n      hooks:  ...{h[k:k + 60]!r}\n      edited: ...{e[k:k + 60]!r}")
print(f"\nIdentical answers: {sum(h == e for h, e in zip(hooked, edited))} of {len(questions)}")
