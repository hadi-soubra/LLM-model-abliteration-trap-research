"""GSM8K: the prompt we add to each problem, and checking the model's final answer."""
import re

INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def as_question(problem: str) -> str:
    return problem + INSTRUCTION


def final_answer(text: str) -> str | None:
    """The number inside the last \\boxed{...}; if there is none, the last number in the answer."""
    start = text.rfind("\\boxed{")
    if start >= 0:
        depth, i = 1, start + len("\\boxed{")
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            text = text[start + len("\\boxed{"):i - 1]
    numbers = NUMBER.findall(text)
    return numbers[-1].replace(",", "").rstrip(".") if numbers else None


def is_correct(predicted: str | None, gold: str) -> bool:
    try:
        return predicted is not None and abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return False
