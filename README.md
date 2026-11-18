# Defending Open-Weight LLMs Against Abliteration

COE749 project (LAU). The goal and plan are in `CLAUDE.md`; the research notebook is in `report/` (local only).

## Setup

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install torch --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=.venv uv pip install -e .
```

## Layout

```
configs/laptop_qwen05b.yaml   settings: model, device, batch size, prompt counts, seed
src/shared/                   config, model loading and chat template, generation, prompts
src/abliteration/             activations, refusal direction, removal (live and in the weights), scoring
steps/                        one script per experiment step, run in order
tests/                        checks for the math
tools/build_notebook.py       copies LEARNING_LOG.md into the notebook
data/                         prompts, activations, directions, chosen layer (gitignored)
report/                       the research notebook (gitignored: it holds raw model outputs)
```

The first version of the code is kept on the git branch `first-pass`.

## Experiment 1: abliteration on the laptop (Qwen2.5-0.5B)

```bash
.venv/bin/python steps/step1_get_prompts.py          # harmful (AdvBench) and harmless (Alpaca) prompts -> data/prompts.json
.venv/bin/python steps/step2_record_activations.py   # internal state at every layer -> data/activations.pt
.venv/bin/python steps/step3_find_direction.py       # one refusal direction per layer -> data/directions.pt
.venv/bin/python steps/step4_choose_layer.py         # score every direction, choose one -> data/chosen_layer.json
.venv/bin/python steps/step5_remove_live.py          # remove it while the model runs, answers before and after
.venv/bin/python steps/step6_edit_weights.py         # bake the removal into the weights, compare with step 5
.venv/bin/python steps/step7_chat.py                 # chat: http://localhost:8000/exp1-local.html#chat
```

Checks for the math: `.venv/bin/python tests/test_direction.py`, `tests/test_ablation.py`, `tests/test_weights.py`.

Abliterated weights are never saved: every edit lives in memory only.

## Notebook

Open `report/index.html` in a browser, or run step 7 and use `http://localhost:8000`.
After adding to `LEARNING_LOG.md`, run `.venv/bin/python tools/build_notebook.py`.
