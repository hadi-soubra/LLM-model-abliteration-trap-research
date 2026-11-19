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


def attention_of(layer):
    """A layer's attention part. Qwen3.5 has two kinds: full attention ("self_attn") in some layers and
    linear attention ("linear_attn") in the others. Both add their output to the running state."""
    return layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn


def attention_writer(layer):
    """The matrix that writes the attention part's output into the running state ("o_proj" or "out_proj")."""
    attention = attention_of(layer)
    return attention.o_proj if hasattr(attention, "o_proj") else attention.out_proj


def chat_text(tokenizer, messages: list[dict]) -> str:
    """A conversation in the model's chat template, ending where the assistant's answer begins.
    Thinking is turned off: Qwen3.5 would otherwise write out long reasoning before every answer."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def wrap(tokenizer, question: str) -> str:
    """Put a single question into the chat template."""
    return chat_text(tokenizer, [{"role": "user", "content": question}])


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
def chat_reply(model, tokenizer, messages: list[dict], max_new_tokens: int = 256,
               repetition_penalty: float = 1.0) -> str:
    """One greedy answer to a whole conversation: a list of {"role": "user" or "assistant", "content": ...}."""
    batch = tokenizer([chat_text(tokenizer, messages)], return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         top_k=None, repetition_penalty=repetition_penalty, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def answer_many(model, tokenizer, questions: list[str], max_new_tokens: int, batch_size: int,
                repetition_penalty: float = 1.0) -> list[dict]:
    """Greedy answers to many questions, in batches of similar length. Per question: the answer, how many tokens
    it used, and whether it hit the length limit (then it was cut off)."""
    order = sorted(range(len(questions)), key=lambda i: -len(questions[i]))
    out = [None] * len(questions)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tokenizer([wrap(tokenizer, questions[i]) for i in idx], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
        generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                   top_p=None, top_k=None, repetition_penalty=repetition_penalty,
                                   pad_token_id=tokenizer.pad_token_id)
        stop_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id} | set(
            model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
            else [model.generation_config.eos_token_id])
        for row, i in zip(generated[:, batch["input_ids"].shape[1]:].tolist(), idx):
            n = next((k for k, t in enumerate(row) if t in stop_ids), len(row))
            out[i] = {"answer": tokenizer.decode(row[:n], skip_special_tokens=True).strip(), "n_tokens": n,
                      "cut_off": n >= max_new_tokens}
        print(f"    answered {min(start + batch_size, len(order))} of {len(order)}", flush=True)
    return out
