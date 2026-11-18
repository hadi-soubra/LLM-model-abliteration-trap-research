"""Step 3 math: the refusal direction at every layer (difference of means, Arditi et al. 2024).

YOU WRITE THIS FUNCTION. Check it with:   .venv/bin/python tests/test_direction.py
"""
import torch


def refusal_directions(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """The refusal direction at every layer.

    harmful:  [harmful prompts, layers, numbers per layer]    (recorded in step 2)
    harmless: [harmless prompts, layers, numbers per layer]
    returns:  [layers, numbers per layer]: one row per layer, each an arrow of length 1 pointing
              from the average harmless prompt toward the average harmful prompt.

    Three moves:
      1. average each pile over its prompts            -> two tensors of shape [layers, numbers per layer]
      2. subtract: harmful average minus harmless average
      3. divide each layer's arrow by its own length    -> every row now has length 1

    Useful:  tensor.mean(dim=0)  averages over the first axis (the prompts)
             tensor.norm(dim=-1, keepdim=True)  is the length of each row, shaped so you can divide by it
    """
      # TODO: your code here (about 3 lines)
    harmful_avg = harmful.mean(dim=0)  # Step 1: average over prompts
    harmless_avg = harmless.mean(dim=0)  # Step 1: average over prompts
    direction = harmful_avg - harmless_avg  # Step 2: subtract
    direction /= direction.norm(dim=-1, keepdim=True)  # Step 3: normalize each row to length 1
    return direction


def harmless_levels(harmless: torch.Tensor, r: torch.Tensor) -> list[float]:
    """A1b's target: the average dial reading of harmless prompts along r, at the input of every layer.

    harmless: [harmless prompts, layers, numbers per layer] (recorded in step 2); r: one direction, length 1.
    """
    return (harmless.float() @ r.float()).mean(dim=0).tolist()
