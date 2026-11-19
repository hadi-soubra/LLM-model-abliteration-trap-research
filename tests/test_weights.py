"""Checks for orthogonalize(), on small made-up matrices.

    .venv/bin/python tests/test_weights.py
"""
import sys

import torch

from abliteration.ablation import remove_direction
from abliteration.weights import orthogonalize, orthogonalize_in_place


def unit(v):
    return v / v.norm()


def blind_along_r():
    """after the edit, the matrix can write nothing along r:  r @ W' = 0"""
    torch.manual_seed(0)
    W, r = torch.randn(8, 5), unit(torch.randn(8))
    left = r @ orthogonalize(W, r)
    assert left.abs().max() < 1e-5, f"r @ W' should be 0, largest value is {left.abs().max():.5f}"


def same_as_removing_afterwards():
    """editing the matrix gives the same output as removing r from what the old matrix wrote (step 5's way)"""
    torch.manual_seed(1)
    W, r, x = torch.randn(8, 5), unit(torch.randn(8)), torch.randn(5)
    assert torch.allclose(orthogonalize(W, r) @ x, remove_direction(W @ x, r), atol=1e-5)


def rest_untouched():
    """everything the matrix writes that is not along r stays the same"""
    torch.manual_seed(2)
    W, r = torch.randn(6, 4), torch.tensor([0, 1.0, 0, 0, 0, 0])
    out = orthogonalize(W, r)
    keep = [0, 2, 3, 4, 5]
    assert torch.allclose(out[keep], W[keep]) and out[1].abs().max() < 1e-6


def keeps_shape_and_format():
    """a bf16 matrix comes back as a bf16 matrix of the same shape, and the original is not changed"""
    W, r = torch.randn(8, 5).bfloat16(), unit(torch.randn(8))
    copy = W.clone()
    out = orthogonalize(W, r)
    assert out.shape == W.shape and out.dtype == torch.bfloat16 and torch.equal(W, copy)


def slices_give_same_result():
    """editing in slices (to save memory on big matrices) gives exactly the same matrix, also for a transposed view"""
    torch.manual_seed(3)
    W, r = torch.randn(8, 50), unit(torch.randn(8))
    sliced = W.clone()
    orthogonalize_in_place(sliced, r, slice_size=7)
    assert torch.allclose(sliced, orthogonalize(W, r), atol=1e-6)
    E = torch.randn(50, 8)  # like an embedding: one row per word
    E_edit = E.clone()
    orthogonalize_in_place(E_edit.T, r, slice_size=7)
    assert torch.allclose(E_edit.T, orthogonalize(E.T, r), atol=1e-6), "editing through a transposed view failed"


if __name__ == "__main__":
    failed = 0
    for check in [blind_along_r, same_as_removing_afterwards, rest_untouched, keeps_shape_and_format,
                  slices_give_same_result]:
        try:
            check()
            print(f"PASS  {check.__doc__}")
        except (AssertionError, RuntimeError) as e:
            failed += 1
            print(f"FAIL  {check.__doc__}\n      {e}")
    print("\nAll checks passed." if not failed else f"\n{failed} check(s) failed.")
    sys.exit(1 if failed else 0)
