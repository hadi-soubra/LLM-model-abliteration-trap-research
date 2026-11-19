# Learning log

One entry per session: what was built, the key idea, and any number that surprised me.
The notebook's journal page shows this file (update it with `tools/build_notebook.py`).

## 14–17 Nov 2026: a first pass, then a restart

**What happened:** a first version of the whole pipeline (data, judge, attacks A1 and A1b) was built and run quickly. It moved faster than my understanding, so all its results and datasets were deleted and the work was redone one step at a time. The old code is on the git branch `first-pass`.

**Lessons that carried over:** keyword counting cannot tell garbage or a sidestep from a real harmful answer, so the project needs a judge model, and the judge's own mistakes must be measured too. A small model is fragile: the standard side-effect limit for choosing a direction (KL below 0.1) can be too strict for it.

## 17 Nov 2026: the attack, one step at a time

**Built:** a smaller codebase that only abliterates (src/shared, src/abliteration, one script per step, tests). I wrote the difference of means and the projection myself; from then on, Claude writes the math.

**Key idea:** the refusal direction is the harmful average minus the harmless average, one arrow per layer. An arrow that separates harmful from harmless prompts does not necessarily control refusal; the test is whether adding it makes the model refuse harmless requests. Removing one controlling arrow everywhere, live or baked into the weights, should turn refusals into compliance.

## 18 Nov 2026: Night 1 prepared

**Built:** A1b (the dial pinned to the harmless level, live and in the weights), the ruler (HarmBench with a judge, the judge's false alarms on harmless requests, XSTest, GSM8K), one night script that runs the whole baseline on the main model, a preflight check, and setup instructions for the RTX 5090 PC. Each model keeps its own data folder, and each run sends back one full zip privately.

**Key idea:** the real runs happen on a friend's PC I cannot reach, so every run must be self-contained: it checks itself, continues after an interruption, and records the code version and environment.

## 19 Nov 2026: the Qwen3.5 family

**Decided:** the main model is Qwen3.5-9B on the RTX 5090, with Qwen3.5-0.8B on the laptop; no fixed deadline. Experiment 1 is redone step by step on the 0.8B.

**Key idea:** Qwen3.5 mixes two kinds of layer, linear attention and full attention, but both add their output to the running state, so abliteration carries over unchanged. Its "thinking" is switched off so every answer is a direct reply.

**Surprising numbers:** Qwen3.5-9B scores 82.5 on MMLU-Pro, at the level of models many times its size. Its vocabulary of about 248,000 words made the embedding so large that the weight edit had to be done in slices to fit in memory. On it, the weight edit matches the live removal to a KL of about 1e-10.

## 19 Nov 2026: Experiment 1 on Qwen3.5-0.8B

**Built:** steps 3 to 7 rerun on the new model (Claude ran steps 4 to 7 on its own at my request). A check of how the model starts its answers (`steps/step4_check_first_words.py`), a relative rule for choosing the arrow, and step 4's scores saved to a file. The chat now decodes like the measurements. Night 1 rehearsed on the 0.8B.

**Key idea:** the original rule for choosing the arrow (adding it must push the refusal score above 0) assumes the model refuses harmful prompts with a confident first word. Qwen3.5 refuses 29 of 32 check prompts with "I cannot…", yet gives "I" only about 65% of the first-word probability, the rest going to markdown openers, so its score on harmful prompts is just 0.40. The rule now asks the arrow to lift harmless prompts halfway to the harmful prompts' score. Layer 13 wins.

**Surprising numbers:** harmless prompts sit near zero on the refusal dial at every layer, so A1 and A1b should behave almost the same on this model. The two piles already separate at layer 2 (86%), long before any refusal decision. In bfloat16 the weight edit gives only 2 of 12 answers identical to the live removal, against 12 of 12 in 32-bit: pure rounding, amplified by greedy decoding.
