"""Step 1: get the prompts the attack needs, and look at them.

    .venv/bin/python steps/step1_get_prompts.py
"""
import json

from shared.config import ROOT, config_from_cli, data_dir
from shared.prompts import download_advbench, download_alpaca, pick_prompts

cfg, _ = config_from_cli()
harmful_all, harmless_all = download_advbench(), download_alpaca()
print(f"AdvBench has {len(harmful_all)} harmful requests; "
      f"Alpaca has {len(harmless_all)} instructions without extra input.")

prompts = pick_prompts(cfg, harmful_all, harmless_all)
for name, items in prompts.items():
    print(f"\n{name}: {len(items)} prompts. The first 3:")
    for text in items[:3]:
        print("  -", text)

out = data_dir(cfg) / "prompts.json"
if out.exists():  # groups that existed before must come out identical
    old = json.loads(out.read_text())
    for name in old:
        print(f"{name}: {'unchanged' if old[name] == prompts.get(name) else 'CHANGED'} since the last run")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(prompts, indent=1))
print(f"\nSaved to {out.relative_to(ROOT)} (nothing else is kept)")
