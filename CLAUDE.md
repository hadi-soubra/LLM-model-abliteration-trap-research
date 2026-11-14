# Project: Defending Open-Weight LLMs Against Abliteration

COE749 Advanced Large Language Models, LAU, Fall 2026 (Dr. Samer Saab Jr).
Deliverable: publication-style paper (Intro, Contributions, Related Work, Method, Experiments, Discussion, Conclusion) plus code.
Implementation budget: ~20 days. Topic approved by the instructor.

## Goal

Two-layer defense for small open-weight chat models:

1. **Hardness:** make training-free abliteration as costly as possible (more directions, more search, more compute needed to remove refusal).
2. **Self-destruct (payoff denial):** if an attacker does remove refusal, the stripped model is useless to them: at minimum its answers to harmful requests become garbage; ideally general capability collapses too. The toy test showed the harmful-answers version is the achievable target against adaptive (affine) attacks.

Framing: we do NOT claim abliteration becomes impossible. Open weights can always be fine-tuned. We claim measurable increases in attacker cost and a measurable drop in the attacker's payoff (useful harmful answers) after a successful attack, while the defended model stays useful on normal tasks.

## Go/no-go gates

- **Day 3:** A1 abliteration reproduced on the chosen model (refusal drops sharply, GSM8K roughly intact). Eval pipeline produces JSON results.
- **Day 6:** minimum trap test on Qwen2.5-1.5B (LoRA, collapse loss on harmful prompts under simulated A1 + A1b ablation, direction recomputed every K steps). PASS if best-attack useful-harmful-answer rate is clearly below the undefended model's and clean GSM8K drops by only a few points. FAIL by day 12 means switching to the fallback paper.

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
- A1b: affine abliteration: reset the direction to its average harmless level instead of zero (one extra constant; can be baked into the embedding matrix). In our toy test this walked straight past a trap trained only against A1, so every trap must be trained and tested against it.
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
- Toy proof of concept (toy_trap_demo.py, small MLP, synthetic data): ordinary model's best attack payoff 73% (useful harmful answers). Trap vs A1 only: A1 payoff 4%, but A1b bypasses it (59%). Trap vs A1 + A1b wrecking all answers: payoff 40%, clean accuracy 83% to 64%. Trap vs A1 + A1b wrecking only harmful answers: best-attack payoff 33%, clean accuracy 83% to 76%. Lessons: train against several attack variants; target the attacker's payoff (harmful answers) rather than all capability, because an affine edit barely changes normal inputs.
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

Candidates: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Gemma-2-2B-it.
First experiment: measure baseline refusal rate on each and pick models that refuse reliably before any attack.

## Data

- Harmful: AdvBench, HarmBench, JailbreakBench
- Harmless: Alpaca-style instructions
- Over-refusal: XSTest
- Keep direction-extraction prompts, defense-training prompts, and evaluation prompts in DISJOINT splits.

## Metrics

- **Refusal / harmfulness:** use a judge model (Llama-Guard-3-1B or a StrongREJECT-style judge), not keyword matching. Keyword matching counts garbled output as "compliant".
- **Generation length:** at least 256 new tokens, so delayed refusals are caught.
- **Capability:** MMLU subset, GSM8K, ARC (before attack, after defense, after attack).
- **Over-refusal:** XSTest safe prompts.
- **Attacker cost:** number of directions, search trials, GPU-minutes needed to reach a target refusal rate.
- **Self-destruct effectiveness:** capability drop of the defended model after a successful attack, plus a coherence check. GSM8K is the most sensitive probe for abliteration damage; use it as the main collapse metric.
- **Refusal-vs-KL frontier:** report minimum refusal at several KL budgets (e.g. 0.05, 0.10, 0.20, 1.0, as in Bait-and-Recover). A KL-budgeted search avoids collapsing edits, so layer 2 only wins if no low-KL edit removes refusal.
- **Repair test:** after a successful attack, fine-tune on a small benign set (10 to 100 examples) and check whether capability comes back.
- Keep precision consistent (BF16) across all compared checkpoints. Quantization/toolchain differences can confound abliteration results.

## Hardware

- Main: RunPod RTX 4090 (24GB), Community Cloud. Stop the pod when idle.
- Backup: Kaggle T4/P100 (no bf16, slower) for long eval runs.
- Local laptop RTX 3060 (6GB): not used for training.

## Engineering conventions

- Python, `transformers`, `peft`, `vllm` for generation, `torch` hooks for activation capture and ablation.
- All runs driven by config files, fixed seeds, results saved as JSON per run (model, defense, attack, metrics).
- Push code to GitHub, checkpoints and results to a PRIVATE Hugging Face repo, so a dead pod loses nothing.
- Log compute (GPU type, hours) for every experiment: the paper needs clear compute reporting.

## Ethics rules (non-negotiable)

- Never publish abliterated weights. All HF repos stay private.
- Do not include harmful model outputs in the paper or logs committed to GitHub. Store raw generations locally/privately only.
- Use standard public safety benchmarks only.

## Timeline (20 days)

- Days 1-3: baselines, reproduce A1 abliteration, eval pipeline working
- Days 4-10: defense training (hardness + self-destruct)
- Days 11-15: full attack suite against all defenses and baselines
- Days 16-17: ablations (each loss term, K for direction recompute, model size)
- Days 18-20: tables, figures, writing
- **Fallback:** if the self-destruct layer isn't working by day 12, pivot to a rigorous evaluation of hardness defenses against adaptive attacks (A1 to A4) with attacker-cost metrics. That is still a complete paper.
