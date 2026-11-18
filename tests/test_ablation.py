"""Checks for remove_direction(), on small made-up numbers where the right answer is known.

    .venv/bin/python tests/test_ablation.py
"""
import sys

import torch

from abliteration.ablation import remove_direction, set_coordinate


def unit(v):
    return v / v.norm()


def dial_reads_zero():
    """after removal, every dial reading h · r is 0"""
    torch.manual_seed(0)
    h, r = torch.randn(3, 5, 8), unit(torch.randn(8))
    readings = remove_direction(h, r) @ r
    assert readings.abs().max() < 1e-5, f"largest dial reading left: {readings.abs().max():.4f}. Expected about 0."


def known_answer():
    """h = 3·r + v, with v at right angles to r  ->  the result is exactly v"""
    r, v = torch.tensor([1.0, 0, 0, 0]), torch.tensor([0, 2.0, -1.0, 5.0])
    out = remove_direction(3 * r + v, r)
    assert torch.allclose(out, v, atol=1e-6), f"got {out.tolist()}, expected {v.tolist()}. Check the sign: h MINUS (h·r) r."


def rest_untouched():
    """everything not along r stays exactly as it was"""
    torch.manual_seed(1)
    h, r = torch.randn(4, 6), torch.tensor([0, 0, 1.0, 0, 0, 0])
    out = remove_direction(h, r)
    keep = [0, 1, 3, 4, 5]
    assert torch.allclose(out[:, keep], h[:, keep], atol=1e-6), "numbers that do not lie along r were changed."


def same_shape_any_batch():
    """works for one list, a batch of prompts, and a batch of prompts x words"""
    r = unit(torch.randn(8))
    for shape in [(8,), (4, 8), (2, 3, 8)]:
        out = remove_direction(torch.randn(*shape), r)
        assert out.shape == shape, (f"input shape {list(shape)} came back as {list(out.shape)}. "
                                    "Use .unsqueeze(-1) on the readings before multiplying by r.")


def twice_is_once():
    """removing a second time changes nothing (there is nothing left along r)"""
    torch.manual_seed(2)
    h, r = torch.randn(5, 8), unit(torch.randn(8))
    once = remove_direction(h, r)
    assert torch.allclose(remove_direction(once, r), once, atol=1e-5), "removing twice gave a different result."


def input_not_changed():
    """h itself is left alone (the hooks rely on this)"""
    torch.manual_seed(3)
    h, r = torch.randn(5, 8), unit(torch.randn(8))
    copy = h.clone()
    remove_direction(h, r)
    assert torch.equal(h, copy), "h was modified in place. Build a new tensor instead of using -= on h."


def a1b_sets_level():
    """A1b: after set_coordinate, every dial reading equals the chosen level, and the rest is untouched"""
    torch.manual_seed(4)
    h, r = torch.randn(3, 5, 8), unit(torch.randn(8))
    out = set_coordinate(h, r, 2.5)
    assert torch.allclose(out @ r, torch.full((3, 5), 2.5), atol=1e-5), "the dial does not read the level"
    assert torch.allclose(remove_direction(out, r), remove_direction(h, r), atol=1e-5), "parts not along r changed"


if __name__ == "__main__":
    failed = 0
    for check in [dial_reads_zero, known_answer, rest_untouched, same_shape_any_batch, twice_is_once, input_not_changed,
                  a1b_sets_level]:
        try:
            check()
            print(f"PASS  {check.__doc__}")
        except NotImplementedError as e:
            sys.exit(f"TODO  {e}")
        except (AssertionError, RuntimeError) as e:
            failed += 1
            print(f"FAIL  {check.__doc__}\n      {e}")
    print("\nAll checks passed." if not failed else f"\n{failed} check(s) failed.")
    sys.exit(1 if failed else 0)
