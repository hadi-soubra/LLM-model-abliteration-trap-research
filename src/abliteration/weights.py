"""Step 6: make the removal permanent by editing the weights (weight orthogonalization, Arditi et al. 2024).

Only the matrices that WRITE into the running list are changed: the embedding, and each layer's attention output
and MLP output. After the edit none of them can write anything along r, so the dial stays at 0 without any hooks.
The edited weights live only in memory; they are never saved.
"""
import torch


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
def edit_weights(model, r: torch.Tensor) -> None:
    """Apply orthogonalize() to every matrix that writes into the running list, in place."""
    r = r.to(model.device)
    embedding, output_layer = model.get_input_embeddings(), model.get_output_embeddings()
    if output_layer.weight.data_ptr() == embedding.weight.data_ptr():
        # In this model the embedding and the final output layer share one matrix. The output layer only READS
        # the running list, so it must not be edited: give it its own copy first.
        output_layer.weight = torch.nn.Parameter(embedding.weight.detach().clone(), requires_grad=False)
        model.config.tie_word_embeddings = False
    # The embedding stores one row per word ([vocabulary, numbers]); transposed, its columns are what it writes.
    embedding.weight.copy_(orthogonalize(embedding.weight.T, r).T)
    for layer in model.model.layers:
        for writer in (layer.self_attn.o_proj, layer.mlp.down_proj):  # attention output, MLP output
            writer.weight.copy_(orthogonalize(writer.weight, r))
            if writer.bias is not None:
                writer.bias.copy_(orthogonalize(writer.bias[:, None], r)[:, 0])
