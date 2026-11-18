"""Load the model, and wrap questions in its chat template."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(cfg: dict):
    """Returns (model, tokenizer). The model is frozen: we only read it, we never train it."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    tokenizer.padding_side = "left"  # pad short prompts on the left, so every prompt ends at the last position
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=DTYPES[cfg["dtype"]]).to(cfg["device"])
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def wrap(tokenizer, question: str) -> str:
    """Put a question into the chat template, ending exactly where the assistant's answer begins."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tokenizer, questions: list[str], max_new_tokens: int = 128) -> list[str]:
    """Greedy answers: at every step the model takes its single most likely next word."""
    texts = [wrap(tokenizer, q) for q in questions]
    batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         top_k=None, repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id)
    new_words = out[:, batch["input_ids"].shape[1]:]  # drop the prompt, keep only the answer
    return [tokenizer.decode(w, skip_special_tokens=True).strip() for w in new_words]


@torch.no_grad()
def chat_reply(model, tokenizer, messages: list[dict], max_new_tokens: int = 256) -> str:
    """One greedy answer to a whole conversation: a list of {"role": "user" or "assistant", "content": ...}."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         top_k=None, repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
