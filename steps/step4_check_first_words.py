"""Step 4 check: how does the normal model start its answers to the harmful check prompts?

The refusal score only counts answers whose first word is "I" or "As". For every harmful check prompt this prints
the score, the model's three most likely first words, and the start of its real answer, so we can see whether a low
score means the model complies, or refuses with other words.

    .venv/bin/python steps/step4_check_first_words.py
"""
import json
from collections import Counter

from abliteration.scoring import first_word_logits, refusal_score, refusal_token_ids
from shared.config import config_from_cli, data_dir
from shared.model import answer_many, load_model

cfg, _ = config_from_cli()
questions = json.loads((data_dir(cfg) / "prompts.json").read_text())["harmful_check"]
model, tokenizer = load_model(cfg)

logits = first_word_logits(model, tokenizer, questions, cfg["batch_size"])
scores = refusal_score(logits, refusal_token_ids(tokenizer))
top = logits.softmax(dim=-1).topk(3)  # the three most likely first words, with their probabilities
answers = answer_many(model, tokenizer, questions, 24, cfg["batch_size"], cfg["decoding"]["repetition_penalty"])

tally = Counter()
for q, s, probs, ids, a in zip(questions, scores, top.values, top.indices, answers):
    words = [tokenizer.decode([t]) for t in ids.tolist()]
    tally[words[0]] += 1
    print(f"score {s:6.2f} | " + "   ".join(f"{w!r} {p:.0%}" for w, p in zip(words, probs.tolist())))
    print(f"   Q: {q[:90]}")
    print(f"   A: {a['answer'][:130]!r}\n")

print(f"Most likely first word, counted over the {len(questions)} prompts:")
for w, n in tally.most_common():
    print(f"  {w!r:>18}  {n}")
print(f"\nAverage refusal score: {scores.mean():.2f}")
