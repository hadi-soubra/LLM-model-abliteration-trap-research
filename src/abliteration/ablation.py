"""Remove the refusal direction (A1) or pin it to its harmless level (A1b) while the model runs.

remove_direction()   A1's edit: the dial reading becomes 0. (Written by Hadi.)
set_coordinate()     A1b's edit: the dial reading becomes a chosen level.
removing()           attaches hooks so the edit is applied everywhere inside the model.
"""
from contextlib import contextmanager

import torch

from shared.model import attention_of


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


def set_coordinate(h: torch.Tensor, r: torch.Tensor, level: float) -> torch.Tensor:
    """A1b: h - (h · r) r + level · r. Everything else stays; the dial reading becomes exactly `level`."""
    return remove_direction(h, r) + level * r


@contextmanager
def removing(model, r: torch.Tensor, levels: list[float] | None = None):
    """Inside `with removing(model, r):` the model runs with direction r removed at every layer (A1).
    With levels (one per layer), the dial is set to levels[i] as it enters layer i instead (A1b).
    What each layer's attention and MLP parts add is always cleaned of r, so nothing can move the dial."""
    r = r.to(model.device, model.dtype)
    handles = []

    def on_input(i):  # just before layer i: clean (or set) what enters it
        def edit(h):
            return remove_direction(h, r) if levels is None else set_coordinate(h, r, float(levels[i]))

        def hook(module, args, kwargs):
            if args:
                return (edit(args[0]),) + tuple(args[1:]), kwargs
            return args, {**kwargs, "hidden_states": edit(kwargs["hidden_states"])}
        return hook

    def on_output(module, args, output):  # just after attention or MLP: clean what it adds
        if isinstance(output, tuple):
            return (remove_direction(output[0], r),) + tuple(output[1:])
        return remove_direction(output, r)

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(on_input(i), with_kwargs=True))
        handles.append(attention_of(layer).register_forward_hook(on_output))
        handles.append(layer.mlp.register_forward_hook(on_output))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()  # always detach, so the model is back to normal afterwards
