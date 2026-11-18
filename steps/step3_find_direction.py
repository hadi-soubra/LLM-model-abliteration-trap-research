"""Step 3: find the refusal direction at every layer, and see where harmful and harmless prompts sit apart.

    .venv/bin/python steps/step3_find_direction.py
"""
import torch

from abliteration.direction import refusal_directions
from shared.config import ROOT

acts = torch.load(ROOT / "data" / "activations.pt")
harmful, harmless = acts["harmful"], acts["harmless"]
r = refusal_directions(harmful, harmless)  # your function: [layers, numbers per layer]
print(f"directions: shape {list(r.shape)}, lengths from {r.norm(dim=-1).min():.3f} to {r.norm(dim=-1).max():.3f}\n")

# each prompt's position on the "dial" at each layer: the dot product of its numbers with that layer's direction
dial_harmful = torch.einsum("pln,ln->pl", harmful, r)    # [prompts, layers]
dial_harmless = torch.einsum("pln,ln->pl", harmless, r)
size = torch.cat([harmful, harmless]).norm(dim=-1).mean(dim=0)  # typical size at each layer (from step 2)

print("layer   harmful avg   harmless avg     gap   gap / size   prompts on their side")
for L in range(r.shape[0]):
    h, b = dial_harmful[:, L], dial_harmless[:, L]
    middle = (h.mean() + b.mean()) / 2
    on_side = ((h > middle).float().mean() + (b < middle).float().mean()) / 2
    print(f"{L:5d}   {h.mean():11.2f}   {b.mean():12.2f}   {h.mean() - b.mean():5.2f}   "
          f"{(h.mean() - b.mean()) / size[L]:10.2f}   {on_side:21.0%}")

torch.save(r, ROOT / "data" / "directions.pt")
print("\nSaved to data/directions.pt")
