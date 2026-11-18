"""Checks for refusal_directions(), on tiny made-up numbers where the right answer is known.

    .venv/bin/python tests/test_direction.py
"""
import sys

import torch

from abliteration.direction import refusal_directions


def shape():
    """one direction per layer: shape [layers, numbers per layer]"""
    harmful, harmless = torch.randn(5, 3, 4), torch.randn(7, 3, 4)  # 5 vs 7 prompts, 3 layers, 4 numbers
    r = refusal_directions(harmful, harmless)
    assert r.shape == (3, 4), (f"expected [3, 4] (layers, numbers), got {list(r.shape)}. "
                               "Average over dim=0, the prompts, so that axis disappears.")


def length_one():
    """every direction has length 1"""
    r = refusal_directions(torch.randn(5, 3, 4) * 10, torch.randn(7, 3, 4))
    lengths = r.norm(dim=-1)
    assert torch.allclose(lengths, torch.ones(3), atol=1e-5), (
        f"lengths per layer are {lengths.tolist()}. Divide each row by its own length: norm(dim=-1, keepdim=True).")


def known_answer():
    """harmful at [3,0,0,0], harmless at [1,0,0,0]  ->  direction [1,0,0,0]"""
    harmful = torch.tensor([3.0, 0, 0, 0]).expand(6, 2, 4)
    harmless = torch.tensor([1.0, 0, 0, 0]).expand(6, 2, 4)
    r = refusal_directions(harmful, harmless)
    assert torch.allclose(r, torch.tensor([[1.0, 0, 0, 0]] * 2), atol=1e-5), (
        f"got {r.tolist()}. If you got [-1, 0, 0, 0], the subtraction is the wrong way round: harmful minus harmless.")


def per_layer():
    """each layer gets its own direction: averages are over prompts, never over layers"""
    harmful = torch.zeros(4, 2, 2)
    harmful[:, 0] = torch.tensor([2.0, 0])   # at layer 0, harmful prompts sit along the first axis
    harmful[:, 1] = torch.tensor([0, 2.0])   # at layer 1, along the second axis
    r = refusal_directions(harmful, torch.zeros(4, 2, 2))
    assert torch.allclose(r, torch.tensor([[1.0, 0], [0, 1.0]]), atol=1e-5), (
        f"got {r.tolist()}, expected [[1, 0], [0, 1]]. Make sure you average over dim=0 (prompts) only.")


def points_toward_harmful():
    """on random data, the direction is exactly parallel to (harmful average - harmless average)"""
    torch.manual_seed(0)
    harmful, harmless = torch.randn(50, 3, 8) + 2.0, torch.randn(40, 3, 8)
    r = refusal_directions(harmful, harmless)
    gap = harmful.mean(0) - harmless.mean(0)
    cosine = (r * gap).sum(-1) / gap.norm(dim=-1)
    assert torch.allclose(cosine, torch.ones(3), atol=1e-5), f"cosine with the gap is {cosine.tolist()}, expected 1."


if __name__ == "__main__":
    failed = 0
    for check in [shape, length_one, known_answer, per_layer, points_toward_harmful]:
        try:
            check()
            print(f"PASS  {check.__doc__}")
        except NotImplementedError as e:
            sys.exit(f"TODO  {e}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {check.__doc__}\n      {e}")
    print("\nAll checks passed." if not failed else f"\n{failed} check(s) failed.")
    sys.exit(1 if failed else 0)
