# Project: Defending Open-Weight LLMs Against Abliteration

COE749 Advanced Large Language Models, LAU, Fall 2026 (Dr. Samer Saab Jr).
Deliverable: publication-style paper (Intro, Contributions, Related Work, Method, Experiments, Discussion, Conclusion) plus code.
Implementation budget: ~20 days. Topic approved by the instructor.

## Goal

Two-layer defense for small open-weight chat models:

1. **Hardness:** make training-free abliteration as costly as possible (more directions, more search, more compute needed to remove refusal).
2. **Self-destruct (payoff denial):** if an attacker does remove refusal, the stripped model is useless to them: at minimum its answers to harmful requests become garbage; ideally general capability collapses too. The harmful-answers version is the primary target; general collapse is a stretch goal.

Framing: we do NOT claim abliteration becomes impossible. Open weights can always be fine-tuned. We claim measurable increases in attacker cost and a measurable drop in the attacker's payoff (useful harmful answers) after a successful attack, while the defended model stays useful on normal tasks.

## DEADLINES: presentation Nov 24, 2026 (results frozen Nov 21); final paper mid-December 2026. Reduced scope for the presentation, this overrides the rest of this file. After Nov 24, extra experiments for the paper (A3, baselines, repair test) are allowed.

- ONE main model: Qwen2.5-7B-Instruct, run overnight on the RTX 5090 (see Hardware). Develop and debug on Qwen2.5-0.5B-Instruct on the laptop first (same family, same code).
- Layer 2 (trap) ONLY. Layer 1 hardness is cited as combinable future work.
- Baselines: undefended model; ablation "trap trained vs A1 only".
- Attacks: trap v1 is trained against A1 + A1b and tested against A1, A1b, A2 (3 rounds) and A4; A3 (Heretic) and A5 if time. Every attack also runs on the undefended model in the same batch. Attacks that get through can go into training for a trap v2.
- Schedule (current plan; the journal page of the notebook tracks status):
  - Nov 14-17: Experiment 1, A1 rebuilt step by step on 0.5B (proof of concept). Done.
  - Next: Experiment 2, Qwen2.5-7B + the ruler (judge, GSM8K, XSTest) + baseline: no attack, A1, A1b. Overnight on the 5090.
  - Then: Experiment 3, the trap: developed on 0.5B on the laptop, trained on 7B overnight. GO/NO-GO on its first result: PASS if best-attack useful-harmful-answer rate is clearly below undefended and clean GSM8K drops only a few points. FAIL -> fallback paper (see Fallback).
  - Then: attacks against the trap (A1, A1b, A2, A4; A3, A5 if time).
  - Nov 21: FREEZE results, make figures/tables. Nov 22-23: slides + rehearsal. Nov 24: present.
  - Nov 25 to mid-Dec: extra experiments for the paper (A3, A5, extended-refusal/ART baselines, repair test, a second model size), then write the paper.

## Working style (IMPORTANT: Hadi is learning, not vibe coding)

- Before writing any component, explain in plain words what it does and why, and wait for "go".
- Small steps: one function or file at a time, at most ~60 lines per step. No big multi-file dumps.
- Laptop steps: Hadi runs each one himself and we read the output together. GPU-PC runs: batched into one overnight run script that Hadi's friend starts; we read the results together the next day.
- After each step, give Hadi one thing to verify himself (a command to run, a number to check, a line to read) and say what result to expect.
- Claude writes all code, including the core math (Hadi's decision, 17 Nov; he wrote the difference of means and the projection himself before that). Explain the math in plain words and give a test he can run.
- Every component gets a sanity test he can run, e.g.: the direction has norm 1; after ablation, `h @ r` is ~0; orthogonalized weights give the same output as hook ablation; refusal rate on harmful prompts is high before any attack.
- Keep a running LEARNING_LOG.md: per session, what was built, the key idea in 2-3 sentences, and any number that surprised us.
- Explanations, results and figures go in the local notebook (report/, a multi-page book); all tracking (plan, status, decisions) only in its journal page.
- When something fails, explain the cause before fixing it.

## Background: how abliteration works

1. Run N harmful and N harmless prompts through the model, record residual stream activations at layer l (last token positions).
2. Refusal direction: `r = mean(harmful) - mean(harmless)`, normalized. Pick the layer/position that best removes refusal on a validation split.
3. Weight orthogonalization: for every matrix that writes to the residual stream (embeddings, attention out proj, MLP down proj): `W' = W - r r^T W`.
4. Inference-time variant (directional ablation): `h' = h - r r^T h` at every layer.

Reference: Arditi et al. 2024, "Refusal in Language Models Is Mediated by a Single Direction".

## Threat model

- **In scope:** white-box attacker with full weights, using training-free, direction-based weight edits. The attacker recomputes directions on the DEFENDED model, not the original.
- **Out of scope:** harmful fine-tuning attacks, prompt-level jailbreaks. Mention as limitations.

## Attack suite (evaluate every defense against all of these)

- A1: single-direction abliteration (Arditi et al.)
- A1b: affine abliteration: reset the direction to its average harmless level instead of zero (one extra constant; can be baked into the embedding matrix). An attacker who knows the trap reacts to the direction sitting at zero can choose another value, so every trap must be trained and tested against it.
- A2: multi-direction abliteration (SVD / multiple diff-of-means directions)
- A3: automated search over layers and strengths (Heretic-style)
- A4: "cleaned" direction attack that removes capability-overlapping components before ablating (Surgical Refusal Ablation style). This is the main adaptive attack against the self-destruct layer.
- A5 (optional): attacks before vs after quantization

## Defense design (initial sketch, update as decisions are made)

- LoRA fine-tuning on the base chat model.
- **Layer 1 (hardness) is NOT novel.** Build it on an existing method: Fail-Closed Alignment (Coalson et al. 2026, arXiv 2602.16977: 10 rounds of find direction, ablate it plus all earlier directions, retrain to refuse) or ReFAT (Yu et al., ICLR 2025: recompute DIM direction every 4 steps, ablate on 50% of harmful inputs). DeepRefusal (EMNLP Findings 2025) is a third variant. Use diverse refusal openings in the data (repetitive "I'm sorry" openings make refusal one-dimensional).
- **Layer 2 (self-destruct) is the contribution.** Template: CTRAP (Yi et al., ACL 2026) simulates one harmful fine-tuning step, then applies a collapse loss at the moved weights. We do the same with the abliteration projection as the simulated step: ablate the current refusal direction(s), then push the model to degrade on benign data in that ablated state.
- Loss terms:
  - Utility retention on benign data (LM loss or KL to the original model)
  - Refusal on harmful prompts, including under simulated ablation (layer 1)
  - Collapse term on benign data in the ablated state (layer 2)
- Recompute refusal direction(s) on the current model every K steps, otherwise the model just moves refusal to a new direction and the trap is avoidable. Note DeepRefusal found recomputing unstable; ReFAT and Fail-Closed made it work.
- Design principles: train against several attack variants; target the attacker's payoff (harmful answers) rather than all capability, because an affine edit barely changes normal inputs.
- Known threats to layer 2: capability-cleaned directions (SRA), other independent refusal directions (concept cones), KL-budgeted search that avoids collapsing edits, and repair by a small benign fine-tune (Lucki et al. 2025 show "removed" knowledge returns after 10 examples).

## Baselines

- Undefended model
- Extended-refusal fine-tuning (Abu Shairah et al. 2025)
- ART, Abliteration-Resistant Tuning (Kuo et al. 2026, arXiv 2605.26526): simulates abliteration in the training loop, but its goal is keeping refusal, not collapsing capability.
- An ablation-trained hardness baseline: Fail-Closed Alignment or ReFAT (whichever has usable code).
- Others (AMRA, DDO, Bait-and-Recover) only if public code exists (Bait-and-Recover: github.com/SparkShieldLab/bait-and-recover)

## Related work to position against

- SEAM (ICLR 2026): self-destruct, but triggered by fine-tuning; its own paper reports weight orthogonalization still succeeds (ASR 0.98), and SEAM + extended refusal ("SEAM-ER") only lowers it to 0.36.
- CTRAP (ACL 2026): collapse trap triggered by harmful fine-tuning, never tested against abliteration. Our layer-2 template.
- Fool's Gold (Russinovich 2026): concedes the strip and poisons answers, but chem/bio only, 9B+ models, hand-authored decoys.
- Surgical Refusal Ablation (Cristofano 2026): basis for attack A4.

## Models

- Main: Qwen2.5-7B-Instruct (ungated, same family as the development model).
- Development: Qwen2.5-0.5B-Instruct on the laptop.
- Optional second size for the paper: Qwen2.5-1.5B or 3B.

## Data

- Harmful: AdvBench, HarmBench, JailbreakBench
- Harmless: Alpaca-style instructions
- Over-refusal: XSTest
- Keep direction-extraction prompts, defense-training prompts, and evaluation prompts in DISJOINT splits.

## Metrics

- **Refusal / harmfulness:** use a judge model, not keyword matching: Llama-Guard-3-8B (Llama-Guard-3-1B as fallback), with the judge's false-alarm rate on harmless answers reported next to every result; a StrongREJECT-style usefulness judge as an option. Keyword matching counts garbled output as "compliant".
- **Decoding:** greedy with repetition penalty 1.05, identical for every model and attack.
- **Generation length:** at least 256 new tokens, so delayed refusals are caught.
- **Capability:** MMLU subset, GSM8K, ARC (before attack, after defense, after attack).
- **Over-refusal:** XSTest safe prompts.
- **Attacker cost:** number of directions, search trials, GPU-minutes needed to reach a target refusal rate.
- **Self-destruct effectiveness:** capability drop of the defended model after a successful attack, plus a coherence check. GSM8K is the most sensitive probe for abliteration damage; use it as the main collapse metric.
- **Refusal-vs-KL frontier:** report minimum refusal at several KL budgets (e.g. 0.05, 0.10, 0.20, 1.0, as in Bait-and-Recover). A KL-budgeted search avoids collapsing edits, so layer 2 only wins if no low-KL edit removes refusal.
- **Repair test:** after a successful attack, fine-tune on a small benign set (10 to 100 examples) and check whether capability comes back.
- Keep precision consistent (BF16) across all compared checkpoints. Quantization/toolchain differences can confound abliteration results.

## Hardware

- Main: a friend's Windows PC with an RTX 5090 (32GB). Hadi has no direct access: runs are prepared as self-contained overnight scripts that the friend starts. Raw generations stay on that PC; only numbers and charts are sent back.
- Laptop RTX 3050 (6GB): development, debugging, small-model work (0.5B), the notebook.
- Backup: RunPod RTX 4090 (24GB) or Kaggle.

## Engineering conventions

- Python, `transformers` (generation; `vllm` optional), `peft`, `torch` hooks for activation capture and ablation.
- All runs driven by config files, fixed seeds, results saved as JSON per run (model, defense, attack, metrics).
- Push code to GitHub (public repo: code only, never data/ or report/), checkpoints and results to a PRIVATE Hugging Face repo.
- Log compute (GPU type, hours) for every experiment: the paper needs clear compute reporting.

## Ethics rules (non-negotiable)

- Never publish abliterated weights. All HF repos stay private.
- Do not include harmful model outputs in the paper or logs committed to GitHub. Store raw generations locally/privately only.
- Use standard public safety benchmarks only.

## Fallback

- If the trap isn't working after its first results, pivot to a rigorous evaluation of existing (hardness) defenses against adaptive attacks (A1 to A4) with attacker-cost metrics. That is still a complete paper.
