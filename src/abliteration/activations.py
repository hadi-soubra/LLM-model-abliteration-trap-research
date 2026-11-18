"""Record what the model computes inside: the residual stream at the last word of each prompt."""
import torch

from shared.model import wrap


@torch.no_grad()
def record_last_word(model, tokenizer, questions: list[str], batch_size: int) -> torch.Tensor:
    """For each question, the numbers entering every layer at the last word of the wrapped prompt.

    Returns a tensor of shape [number of questions, number of layers, numbers per layer] in float32.
    """
    layers = model.model.layers
    notes = {}  # layer index -> what entered that layer, for the batch being processed

    def make_hook(i):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]  # [batch, words, numbers per layer]
            notes[i] = hidden[:, -1, :].float().cpu()             # keep only the last word
        return hook

    # a "pre-hook" runs just before the layer does its work, so it sees what flows INTO the layer
    handles = [layer.register_forward_pre_hook(make_hook(i), with_kwargs=True) for i, layer in enumerate(layers)]
    recorded = []
    try:
        for start in range(0, len(questions), batch_size):
            texts = [wrap(tokenizer, q) for q in questions[start:start + batch_size]]
            batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            model(**batch, use_cache=False, logits_to_keep=1)  # run the model; the hooks take the notes
            recorded.append(torch.stack([notes[i] for i in range(len(layers))], dim=1))
    finally:
        for h in handles:
            h.remove()  # always detach the hooks, so the model runs normally afterwards
    return torch.cat(recorded)
