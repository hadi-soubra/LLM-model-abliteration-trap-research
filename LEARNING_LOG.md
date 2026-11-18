# Learning log

One entry per session: what was built, the key idea, and any number that surprised me.
The notebook's journal page shows this file (update it with `tools/build_notebook.py`).

## 14–17 Nov 2026: a first pass, then a restart

**What happened:** a first version of the whole pipeline (data, judge, attacks A1 and A1b) was built and run quickly. It moved faster than my understanding, so all its results and datasets were deleted and the work was redone one step at a time. The old code is on the git branch `first-pass`.

**Lessons that carried over:** keyword counting cannot tell garbage or a sidestep from a real harmful answer, so the project needs a judge model, and the judge's own mistakes must be measured too. A small model is fragile: the standard side-effect limit for choosing a direction (KL below 0.1) is too strict for it.

## 17 Nov 2026: attack A1 rebuilt, one step at a time

**Built:** a smaller codebase that only abliterates (src/shared, src/abliteration, steps 1 to 6, tests). I wrote the difference of means and the projection myself; from step 6 on, Claude writes the math.

**Key idea:** the refusal direction is the harmful average minus the harmless average, one arrow per layer. Many arrows separate harmful from harmless prompts, but only those from layer 13 on actually control refusal: adding them makes the model refuse harmless requests. Removing one such arrow everywhere, live or baked into the weights, turned ten refusals into ten attempts to comply.

**Surprising numbers:** layers 2 to 12 separate the two piles 90 to 97% of the time yet do not control refusal. Removing the layer-1 arrow gave the lowest refusal score of all only because it breaks the model (side effect 6.1). No arrow met the side-effect limit of 0.1; layer 15 was chosen at 0.225.

## 17 Nov 2026: Experiment 1 closed

**Built:** step 7, a chat with the normal and the abliterated model side by side; the notebook brought up to date with the new code.

**Key idea:** Experiment 1 is a proof of concept, judged by eye. A2 to A5 only matter against a trap, and the real test needs a capable model, so the next experiment moves to Qwen2.5-1.5B and builds the ruler: a judge for harmful answers and math problems for ability.
