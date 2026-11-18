"""Step 4: judge a candidate arrow quickly, by looking only at the first word of the model's answer.

Refusals start with "I" ("I'm sorry...") or "As" ("As an AI..."), so how likely the first word is one of those
tells us whether the model is about to refuse, without generating whole answers.
"""
from contextlib import contextmanager

import torch

from shared.model import wrap

REFUSAL_WORDS = ("I", "As")  # the refusal openings Arditi et al. use for Qwen models


def refusal_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer.encode(w, add_special_tokens=False) for w in REFUSAL_WORDS]
    assert all(len(i) == 1 for i in ids), "each refusal word must be a single token"
    return [i[0] for i in ids]


@torch.no_grad()
def first_word_logits(model, tokenizer, questions: list[str], batch_size: int) -> torch.Tensor:
    """The model's raw score for every possible first word of its answer: [questions, vocabulary]."""
    out = []
    for start in range(0, len(questions), batch_size):
        texts = [wrap(tokenizer, q) for q in questions[start:start + batch_size]]
        batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out.append(model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float())
    return torch.cat(out)


def refusal_score(logits: torch.Tensor, refusal_ids: list[int]) -> torch.Tensor:
    """Per question: log(p / (1 - p)), where p = probability that the first word is a refusal word.
    Positive = leans toward refusing, negative = leans toward answering, 0 = a coin flip."""
    p = logits.double().softmax(dim=-1)[:, refusal_ids].sum(dim=-1)
    return torch.log(p + 1e-8) - torch.log(1 - p + 1e-8)


def kl_divergence(logits_before: torch.Tensor, logits_after: torch.Tensor) -> torch.Tensor:
    """Per question: how different two sets of first-word probabilities are. 0 means identical."""
    p, q = logits_before.double().log_softmax(dim=-1), logits_after.double().log_softmax(dim=-1)
    return (p.exp() * (p - q)).sum(dim=-1)


@contextmanager
def adding(model, layer: int, v: torch.Tensor):
    """Inside the block, v is added to everything entering one layer, for every word."""
    v = v.to(model.device, model.dtype)

    def hook(module, args, kwargs):
        if args:
            return (args[0] + v,) + tuple(args[1:]), kwargs
        return args, {**kwargs, "hidden_states": kwargs["hidden_states"] + v}

    handle = model.model.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
