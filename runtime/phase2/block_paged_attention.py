"""Transformers attention hook for BlockPagedCache + fused paged decode."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from runtime.phase2.paged_attention_triton import (
    get_paged_decode_batch_metadata,
    paged_attention_decode_batched,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def block_paged_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Attention for BlockPagedCache: fused paged decode when q_len=1, else eager on gathered K/V."""
    num_kv_groups = getattr(module, "num_key_value_groups", 1)
    meta = get_paged_decode_batch_metadata()
    if meta is not None and query.shape[-2] == 1 and query.is_cuda:
        # Always use paged_attention_decode_batched — it has a torch fallback when
        # Triton is unavailable. Gathering full K/V every layer scales launches
        # with batch size and defeats continuous-batch amortization.
        attn_output = paged_attention_decode_batched(
            query=query,
            metadata=meta,
            num_kv_groups=num_kv_groups,
            scaling=scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    key_states = repeat_kv(key, num_kv_groups)
    value_states = repeat_kv(value, num_kv_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and module.training:
        attn_weights = F.dropout(attn_weights, p=dropout)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
