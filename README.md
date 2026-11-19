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
configs/                      one settings file per model: laptop_qwen35_08b.yaml, gpu_qwen35_9b.yaml
src/shared/                   config, model loading and chat template, generation, prompts
src/abliteration/             activations, refusal direction, A1 and A1b (live and in the weights), scoring
src/evaluation/               the ruler: test sets, judge, GSM8K scoring, summaries
steps/                        Experiment 1 steps, and the overnight runs (night1_baseline.py)
tests/                        checks for the math
tools/                        preflight.py (is this computer ready?), pack_results.py (rebuild the zips), build_notebook.py
GPU_PC_INSTRUCTIONS.md        setup and run instructions for the RTX 5090 PC
data/<config>/                per model: prompts, activations, directions, chosen layer (gitignored)
results/<config>/             per night run: summary, logs, raw answers, environment (gitignored; sent back privately as a full zip)
report/                       the research notebook (gitignored: it holds raw model outputs)
```

The first version of the code is kept on the git branch `first-pass`.

## Experiment 1: abliteration on the laptop (Qwen3.5-0.8B)

```bash
.venv/bin/python steps/step1_get_prompts.py          # harmful (AdvBench) and harmless (Alpaca) prompts -> data/<config>/prompts.json
.venv/bin/python steps/step2_record_activations.py   # internal state at every layer -> data/<config>/activations.pt
.venv/bin/python steps/step3_find_direction.py       # one refusal direction per layer -> data/<config>/directions.pt
.venv/bin/python steps/step4_choose_layer.py         # score every direction, choose one -> chosen_layer.json, step4_scores.json
.venv/bin/python steps/step4_check_first_words.py   # optional: how the normal model starts its answers to harmful prompts
.venv/bin/python steps/step5_remove_live.py          # remove it while the model runs, answers before and after
.venv/bin/python steps/step6_edit_weights.py         # bake the removal into the weights, compare with step 5
.venv/bin/python steps/step7_chat.py                 # chat: http://localhost:8000/exp1-local.html#chat
```

Every step takes `--config` (default `configs/laptop_qwen35_08b.yaml`).

Checks for the math: `.venv/bin/python tests/test_direction.py`, `tests/test_ablation.py`, `tests/test_weights.py`.

Abliterated weights are never saved: every edit lives in memory only.

## Experiment 2: the baseline (overnight on the RTX 5090)

See `GPU_PC_INSTRUCTIONS.md` for the Windows setup. One command runs steps 1 to 4 on Qwen3.5-9B, then the ruler on
the model with no attack, A1 and A1b, and writes `SEND_TO_HADI_gpu_qwen35_9b_night1_full.zip` (everything; send it privately):

```bash
python steps/night1_baseline.py --config configs/gpu_qwen35_9b.yaml
python steps/night1_baseline.py --quick        # laptop rehearsal: 4 questions per test set
```

## Notebook

Open `report/index.html` in a browser, or run step 7 and use `http://localhost:8000`.
After adding to `LEARNING_LOG.md`, run `.venv/bin/python tools/build_notebook.py`.
