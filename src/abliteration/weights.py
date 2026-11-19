"""Step 6: make the removal permanent by editing the weights (weight orthogonalization, Arditi et al. 2024).

Only the matrices that WRITE into the running list are changed: the embedding, and each layer's attention output
and MLP output. After the edit none of them can write anything along r, so the dial stays at 0 without any hooks.
The edited weights live only in memory; they are never saved.
"""
import torch

from shared.model import attention_writer


def orthogonalize(W: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """W - r (rᵀ W): remove the r-part from everything the matrix W can write.

    W: [numbers per layer, inputs]   each column is something W can add to the running list
    r: [numbers per layer]           length 1
    returns a new matrix of the same shape and type, with r @ result equal to 0 (it is blind along r).
    Computed in float32 and stored back in W's own format (bf16), to keep rounding errors small.
    """
    Wf, rf = W.float(), r.float()
    return (Wf - torch.outer(rf, rf @ Wf)).to(W.dtype)


@torch.no_grad()
def orthogonalize_in_place(W: torch.Tensor, r: torch.Tensor, slice_size: int = 8192) -> None:
    """The same edit as orthogonalize(), applied to W directly, a slice of columns at a time.
    Only one slice is ever converted to float32, so even a huge embedding needs little extra memory."""
    rf = r.float()
    for start in range(0, W.shape[1], slice_size):
        part = W[:, start:start + slice_size].float()
        W[:, start:start + slice_size] = (part - torch.outer(rf, rf @ part)).to(W.dtype)


@torch.no_grad()
def edit_weights(model, r: torch.Tensor, levels: list[float] | None = None) -> None:
    """A1: apply the edit to every matrix that writes into the running list, in place.
    A1b (levels given, one per layer): afterwards, pin the dial to levels[i] at the input of layer i."""
    r = r.to(model.device)
    embedding, output_layer = model.get_input_embeddings(), model.get_output_embeddings()
    if output_layer.weight.data_ptr() == embedding.weight.data_ptr():
        # In this model the embedding and the final output layer share one matrix. The output layer only READS
        # the running list, so it must not be edited: give it its own copy first.
        output_layer.weight = torch.nn.Parameter(embedding.weight.detach().clone(), requires_grad=False)
        model.config.tie_word_embeddings = False
    # The embedding stores one row per word ([vocabulary, numbers]); transposed, its columns are what it writes.
    orthogonalize_in_place(embedding.weight.T, r)
    for layer in model.model.layers:
        for writer in (attention_writer(layer), layer.mlp.down_proj):  # attention output, MLP output
            orthogonalize_in_place(writer.weight, r)
            if writer.bias is not None:
                writer.bias.copy_(orthogonalize(writer.bias[:, None], r)[:, 0])
    if levels is not None:
        _pin_levels(model, r, levels)


@torch.no_grad()
def _pin_levels(model, r: torch.Tensor, levels: list[float]) -> None:
    """After orthogonalizing, nothing writes along r, so the dial keeps whatever value the embedding gives it.
    Adding levels[0]·r to every word's embedding sets it at layer 0; a bias of (levels[i+1] - levels[i])·r on
    layer i's MLP output moves it to the next layer's level. This bakes A1b's hooks into the weights."""
    rf = r.float()
    embedding = model.get_input_embeddings()
    embedding.weight.add_((float(levels[0]) * rf).to(embedding.weight.dtype))
    layers = model.model.layers
    for i in range(len(layers) - 1):
        step = float(levels[i + 1]) - float(levels[i])
        writer = layers[i].mlp.down_proj
        if writer.bias is None:  # Qwen's MLP has no bias; give it one (zeros, then the step along r)
            writer.bias = torch.nn.Parameter(torch.zeros(writer.out_features, dtype=writer.weight.dtype,
                                                         device=writer.weight.device), requires_grad=False)
        writer.bias.add_((step * rf).to(writer.bias.dtype))
