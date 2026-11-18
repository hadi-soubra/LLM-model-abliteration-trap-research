"""Step 4: remove the refusal direction from everything the model computes, while it runs.

remove_direction()  YOU WRITE THIS.   Check it with:   .venv/bin/python tests/test_ablation.py
removing()          already written: attaches hooks so remove_direction() is applied everywhere.
"""
from contextlib import contextmanager

import torch


def remove_direction(h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Remove the part of h that lies along the direction r.

    h:       [..., numbers per layer]   any number of leading axes (prompts, words)
    r:       [numbers per layer]        length 1
    returns: same shape as h, but every dial reading (h · r) is now 0. Do not change h itself.

    The formula:   h - (h · r) r
      h @ r            every dial reading at once                  -> shape [...]
      .unsqueeze(-1)   adds a last axis of size 1                  -> shape [..., 1]
      readings * r     turns each reading back into an arrow       -> shape [..., numbers per layer]
    """
    return h - (h @ r.unsqueeze(-1)) * r

@contextmanager
def removing(model, r: torch.Tensor):
    """Inside `with removing(model, r):` the model runs with direction r removed at every layer:
    from what enters each layer, and from what each layer's attention and MLP parts add."""
    r = r.to(model.device, model.dtype)
    handles = []

    def on_input(module, args, kwargs):  # just before a layer: clean what enters it
        if args:
            return (remove_direction(args[0], r),) + tuple(args[1:]), kwargs
        return args, {**kwargs, "hidden_states": remove_direction(kwargs["hidden_states"], r)}

    def on_output(module, args, output):  # just after attention or MLP: clean what it adds
        if isinstance(output, tuple):
            return (remove_direction(output[0], r),) + tuple(output[1:])
        return remove_direction(output, r)

    for layer in model.model.layers:
        handles.append(layer.register_forward_pre_hook(on_input, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(on_output))
        handles.append(layer.mlp.register_forward_hook(on_output))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()  # always detach, so the model is back to normal afterwards
