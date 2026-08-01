"""Fused paged-attention decode kernel (Triton when available, PyTorch fallback otherwise).

Supports batched decode: ``query [B, heads, 1, dim]`` with ``block_tables [B, max_blocks]``
and ``seq_lens [B]``. See ``docs/adr/0005-paged-attention-kernel.md``.
"""

from __future__ import annotations

import contextvars
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from runtime.phase2.paged_kv_kernel import GpuKVBlockPool

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

_PAGED_ATTN_MAX_SPLITS = int(os.getenv("PAGED_ATTN_MAX_SPLITS", "8"))
_PAGED_ATTN_SPLIT_SEQ_THRESHOLD = int(os.getenv("PAGED_ATTN_SPLIT_SEQ_THRESHOLD", "512"))

_decode_scratch: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_direct_out_scratch: dict[tuple, torch.Tensor] = {}

# Forward-scoped batch metadata (set by BlockPagedLayer.update, read by attention hook).
_decode_batch_metadata: contextvars.ContextVar[PagedDecodeBatchMetadata | None] = contextvars.ContextVar(
    "paged_decode_batch_metadata", default=None
)


@dataclass(frozen=True)
class PagedDecodeBatchMetadata:
    """Batched decode metadata passed explicitly to the kernel."""

    pool: GpuKVBlockPool
    layer_idx: int
    block_tables: torch.Tensor  # [B, max_blocks] int32
    seq_lens: torch.Tensor  # [B] int32 — KV length to attend over (includes current token)


# Back-compat alias
PagedDecodeContext = PagedDecodeBatchMetadata


def set_paged_decode_batch_metadata(meta: PagedDecodeBatchMetadata | None) -> contextvars.Token:
    return _decode_batch_metadata.set(meta)


def reset_paged_decode_batch_metadata(token: contextvars.Token) -> None:
    _decode_batch_metadata.reset(token)


def get_paged_decode_batch_metadata() -> PagedDecodeBatchMetadata | None:
    return _decode_batch_metadata.get()


def set_paged_decode_context(ctx: PagedDecodeBatchMetadata | None) -> None:
    """Deprecated: use set_paged_decode_batch_metadata. Kept for callers migrating off globals."""
    _decode_batch_metadata.set(ctx)


def get_paged_decode_context() -> PagedDecodeBatchMetadata | None:
    return get_paged_decode_batch_metadata()


def triton_available() -> bool:
    return _TRITON_AVAILABLE


def _effective_num_splits(num_logical_blocks: int, seq_len: int) -> int:
    if num_logical_blocks <= 1:
        return 1
    if seq_len < _PAGED_ATTN_SPLIT_SEQ_THRESHOLD:
        return 1
    return max(1, min(_PAGED_ATTN_MAX_SPLITS, num_logical_blocks))


def _decode_scratch_buffers(
    device: torch.device,
    batch_size: int,
    num_heads: int,
    num_splits: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device, batch_size, num_heads, num_splits, head_dim)
    cached = _decode_scratch.get(key)
    if cached is not None:
        return cached
    m_parts = torch.empty((batch_size, num_heads, num_splits), device=device, dtype=torch.float32)
    l_parts = torch.empty((batch_size, num_heads, num_splits), device=device, dtype=torch.float32)
    acc_parts = torch.empty((batch_size, num_heads, num_splits, head_dim), device=device, dtype=torch.float32)
    _decode_scratch[key] = (m_parts, l_parts, acc_parts)
    return m_parts, l_parts, acc_parts


def _direct_out_buffer(device: torch.device, batch_size: int, num_heads: int, head_dim: int) -> torch.Tensor:
    key = (device, batch_size, num_heads, head_dim)
    cached = _direct_out_scratch.get(key)
    if cached is not None:
        return cached
    out = torch.empty((batch_size, num_heads, head_dim), device=device, dtype=torch.float32)
    _direct_out_scratch[key] = out
    return out


def warmup_paged_attention_kernel(
    pool: GpuKVBlockPool,
    *,
    num_heads: int,
    num_kv_groups: int,
    head_dim: int,
    block_ids: list[int],
    seq_len: int = 16,
) -> None:
    if not _TRITON_AVAILABLE:
        return
    device = pool.device
    query = torch.zeros(1, num_heads, 1, head_dim, device=device, dtype=pool.dtype)
    scaling = head_dim**-0.5
    paged_attention_decode(
        query=query,
        pool=pool,
        layer_idx=0,
        block_ids=block_ids,
        seq_len=seq_len,
        num_kv_groups=num_kv_groups,
        scaling=scaling,
        block_table=torch.tensor(block_ids, device=device, dtype=torch.int32),
    )
    if device.type == "cuda":
        torch.cuda.synchronize()


if _TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["max_logical_blocks"])
    def _paged_attn_decode_direct_batched_kernel(
        Q,
        K_cache,
        V_cache,
        BlockTables,
        SeqLens,
        Out,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_bt_b,
        stride_bt_m,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_od,
        num_queries_per_kv: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        BLOCK_D: tl.constexpr,
        max_logical_blocks,
        scale,
    ):
        req_id = tl.program_id(0)
        head_id = tl.program_id(1)
        kv_head_id = head_id // num_queries_per_kv

        d_idx = tl.arange(0, BLOCK_D)
        d_mask = d_idx < head_dim
        t_idx = tl.arange(0, block_size)

        q = tl.load(
            Q + req_id * stride_qb + head_id * stride_qh + d_idx * stride_qd,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)

        seq_len = tl.load(SeqLens + req_id).to(tl.int32)
        num_logical_blocks = (seq_len + block_size - 1) // block_size
        if num_logical_blocks > max_logical_blocks:
            num_logical_blocks = max_logical_blocks

        block_row = BlockTables + req_id * stride_bt_b

        m_i = tl.full([], -float("inf"), tl.float32)
        l_i = tl.full([], 0.0, tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for lb in range(max_logical_blocks):
            if lb < num_logical_blocks:
                physical = tl.load(block_row + lb * stride_bt_m).to(tl.int32)
                base_pos = lb * block_size
                tok_mask = (base_pos + t_idx) < seq_len
                load_mask = tok_mask[:, None] & d_mask[None, :]

                k_ptrs = (
                    K_cache
                    + physical * stride_kb
                    + kv_head_id * stride_kh
                    + t_idx[:, None] * stride_kt
                    + d_idx[None, :] * stride_kd
                )
                v_ptrs = (
                    V_cache
                    + physical * stride_vb
                    + kv_head_id * stride_vh
                    + t_idx[:, None] * stride_vt
                    + d_idx[None, :] * stride_vd
                )
                k_tile = tl.load(k_ptrs, mask=load_mask, other=0.0).to(tl.float32)
                v_tile = tl.load(v_ptrs, mask=load_mask, other=0.0).to(tl.float32)

                scores = tl.sum(q[None, :] * k_tile, axis=1) * scale
                scores = tl.where(tok_mask, scores, -float("inf"))

                m_new = tl.maximum(m_i, tl.max(scores, axis=0))
                p = tl.exp(scores - m_new)
                alpha = tl.exp(m_i - m_new)
                l_i = l_i * alpha + tl.sum(p, axis=0)
                acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
                m_i = m_new

        out = acc / l_i
        tl.store(
            Out + req_id * stride_ob + head_id * stride_oh + d_idx * stride_od,
            out,
            mask=d_mask,
        )

    @triton.jit(do_not_specialize=["seq_len", "num_logical_blocks", "blocks_per_split"])
    def _paged_attn_decode_split_kernel(
        Q,
        K_cache,
        V_cache,
        BlockTable,
        M_parts,
        L_parts,
        Acc_parts,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_mh,
        stride_ms,
        stride_ah,
        stride_as,
        stride_ad,
        num_queries_per_kv: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        BLOCK_D: tl.constexpr,
        seq_len,
        num_logical_blocks,
        blocks_per_split,
        scale,
    ):
        head_id = tl.program_id(0)
        split_id = tl.program_id(1)
        kv_head_id = head_id // num_queries_per_kv

        d_idx = tl.arange(0, BLOCK_D)
        d_mask = d_idx < head_dim
        t_idx = tl.arange(0, block_size)

        q = tl.load(Q + head_id * stride_qh + d_idx * stride_qd, mask=d_mask, other=0.0).to(tl.float32)

        m_i = tl.full([], -float("inf"), tl.float32)
        l_i = tl.full([], 0.0, tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        block_start = split_id * blocks_per_split
        block_end = block_start + blocks_per_split
        if block_end > num_logical_blocks:
            block_end = num_logical_blocks

        for lb in range(block_start, block_end):
            physical = tl.load(BlockTable + lb).to(tl.int32)
            base_pos = lb * block_size
            tok_mask = (base_pos + t_idx) < seq_len
            load_mask = tok_mask[:, None] & d_mask[None, :]

            k_ptrs = (
                K_cache
                + physical * stride_kb
                + kv_head_id * stride_kh
                + t_idx[:, None] * stride_kt
                + d_idx[None, :] * stride_kd
            )
            v_ptrs = (
                V_cache
                + physical * stride_vb
                + kv_head_id * stride_vh
                + t_idx[:, None] * stride_vt
                + d_idx[None, :] * stride_vd
            )
            k_tile = tl.load(k_ptrs, mask=load_mask, other=0.0).to(tl.float32)
            v_tile = tl.load(v_ptrs, mask=load_mask, other=0.0).to(tl.float32)

            scores = tl.sum(q[None, :] * k_tile, axis=1) * scale
            scores = tl.where(tok_mask, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
            m_i = m_new

        tl.store(M_parts + head_id * stride_mh + split_id * stride_ms, m_i)
        tl.store(L_parts + head_id * stride_mh + split_id * stride_ms, l_i)
        tl.store(
            Acc_parts + head_id * stride_ah + split_id * stride_as + d_idx * stride_ad,
            acc,
            mask=d_mask,
        )

    @triton.jit(do_not_specialize=["seq_len", "num_logical_blocks"])
    def _paged_attn_decode_direct_kernel(
        Q,
        K_cache,
        V_cache,
        BlockTable,
        Out,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_oh,
        stride_od,
        num_queries_per_kv: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        BLOCK_D: tl.constexpr,
        seq_len,
        num_logical_blocks,
        scale,
    ):
        head_id = tl.program_id(0)
        kv_head_id = head_id // num_queries_per_kv

        d_idx = tl.arange(0, BLOCK_D)
        d_mask = d_idx < head_dim
        t_idx = tl.arange(0, block_size)

        q = tl.load(Q + head_id * stride_qh + d_idx * stride_qd, mask=d_mask, other=0.0).to(tl.float32)

        m_i = tl.full([], -float("inf"), tl.float32)
        l_i = tl.full([], 0.0, tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for lb in range(num_logical_blocks):
            physical = tl.load(BlockTable + lb).to(tl.int32)
            base_pos = lb * block_size
            tok_mask = (base_pos + t_idx) < seq_len
            load_mask = tok_mask[:, None] & d_mask[None, :]

            k_ptrs = (
                K_cache
                + physical * stride_kb
                + kv_head_id * stride_kh
                + t_idx[:, None] * stride_kt
                + d_idx[None, :] * stride_kd
            )
            v_ptrs = (
                V_cache
                + physical * stride_vb
                + kv_head_id * stride_vh
                + t_idx[:, None] * stride_vt
                + d_idx[None, :] * stride_vd
            )
            k_tile = tl.load(k_ptrs, mask=load_mask, other=0.0).to(tl.float32)
            v_tile = tl.load(v_ptrs, mask=load_mask, other=0.0).to(tl.float32)

            scores = tl.sum(q[None, :] * k_tile, axis=1) * scale
            scores = tl.where(tok_mask, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
            m_i = m_new

        out = acc / l_i
        tl.store(Out + head_id * stride_oh + d_idx * stride_od, out, mask=d_mask)

    @triton.jit
    def _paged_attn_combine_kernel(
        M_parts,
        L_parts,
        Acc_parts,
        Out,
        stride_mh,
        stride_ms,
        stride_lh,
        stride_ls,
        stride_ah,
        stride_as,
        stride_ad,
        stride_oh,
        stride_od,
        num_splits,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        head_id = tl.program_id(0)
        d_idx = tl.arange(0, BLOCK_D)
        d_mask = d_idx < head_dim

        m_global = tl.full([], -float("inf"), tl.float32)
        for s in range(num_splits):
            m_s = tl.load(M_parts + head_id * stride_mh + s * stride_ms)
            m_global = tl.maximum(m_global, m_s)

        l_global = tl.full([], 0.0, tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for s in range(num_splits):
            m_s = tl.load(M_parts + head_id * stride_mh + s * stride_ms)
            l_s = tl.load(L_parts + head_id * stride_lh + s * stride_ls)
            scale = tl.exp(m_s - m_global)
            l_global += l_s * scale
            acc_s = tl.load(
                Acc_parts + head_id * stride_ah + s * stride_as + d_idx * stride_ad,
                mask=d_mask,
                other=0.0,
            )
            acc += acc_s * scale

        out = acc / l_global
        tl.store(Out + head_id * stride_oh + d_idx * stride_od, out, mask=d_mask)


def _paged_attention_decode_batched_triton(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    assert _TRITON_AVAILABLE
    batch_size, num_heads, q_len, head_dim = query.shape
    if q_len != 1:
        raise ValueError("Triton paged decode expects q_len=1")
    block_size = k_cache.shape[2]
    max_logical_blocks = int(block_tables.shape[1])

    if batch_size == 1:
        table = block_tables[0, :max_logical_blocks]
        seq_len = int(seq_lens[0].item())
        num_logical = min(max_logical_blocks, math.ceil(seq_len / block_size))
        return _paged_attention_decode_triton_single(
            query,
            k_cache,
            v_cache,
            table[:num_logical],
            seq_len,
            scaling,
            num_kv_groups,
        )

    max_seq = int(seq_lens.max().item())
    num_splits = _effective_num_splits(max_logical_blocks, max_seq)
    block_d = triton.next_power_of_2(head_dim)

    if num_splits == 1:
        out_buf = _direct_out_buffer(query.device, batch_size, num_heads, head_dim)
        grid = (batch_size, num_heads)
        _paged_attn_decode_direct_batched_kernel[grid](
            query,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            out_buf,
            query.stride(0),
            query.stride(1),
            query.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            out_buf.stride(0),
            out_buf.stride(1),
            out_buf.stride(2),
            num_kv_groups,
            head_dim,
            block_size,
            block_d,
            max_logical_blocks,
            scaling,
            num_warps=4,
        )
        return out_buf.to(query.dtype).unsqueeze(2)

    # Long-seq multi-split: fall back to per-row launches (rare in batched decode).
    outs = []
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        nlog = min(max_logical_blocks, math.ceil(seq_len / block_size))
        row = _paged_attention_decode_triton_single(
            query[b : b + 1],
            k_cache,
            v_cache,
            block_tables[b, :nlog],
            seq_len,
            scaling,
            num_kv_groups,
        )
        outs.append(row)
    return torch.cat(outs, dim=0)


def _paged_attention_decode_triton_single(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    _, num_heads, _, head_dim = query.shape
    block_size = k_cache.shape[2]
    num_logical_blocks = int(block_table.numel())
    num_splits = _effective_num_splits(num_logical_blocks, seq_len)
    block_d = triton.next_power_of_2(head_dim)

    if num_splits == 1:
        out_buf = _direct_out_buffer(query.device, 1, num_heads, head_dim)
        grid = (num_heads,)
        _paged_attn_decode_direct_kernel[grid](
            query,
            k_cache,
            v_cache,
            block_table,
            out_buf[0],
            query.stride(1),
            query.stride(3),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            out_buf.stride(1),
            out_buf.stride(2),
            num_kv_groups,
            head_dim,
            block_size,
            block_d,
            seq_len,
            num_logical_blocks,
            scaling,
            num_warps=4,
        )
        return out_buf.to(query.dtype).unsqueeze(2)

    blocks_per_split = (num_logical_blocks + num_splits - 1) // num_splits
    m_parts, l_parts, acc_parts = _decode_scratch_buffers(query.device, 1, num_heads, num_splits, head_dim)
    out_buf = _direct_out_buffer(query.device, 1, num_heads, head_dim)

    grid = (num_heads, num_splits)
    _paged_attn_decode_split_kernel[grid](
        query,
        k_cache,
        v_cache,
        block_table,
        m_parts[0],
        l_parts[0],
        acc_parts[0],
        query.stride(1),
        query.stride(3),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        m_parts.stride(1),
        m_parts.stride(2),
        acc_parts.stride(1),
        acc_parts.stride(2),
        acc_parts.stride(3),
        num_kv_groups,
        head_dim,
        block_size,
        block_d,
        seq_len,
        num_logical_blocks,
        blocks_per_split,
        scaling,
        num_warps=4,
    )

    _paged_attn_combine_kernel[(num_heads,)](
        m_parts[0],
        l_parts[0],
        acc_parts[0],
        out_buf[0],
        m_parts.stride(1),
        m_parts.stride(2),
        l_parts.stride(1),
        l_parts.stride(2),
        acc_parts.stride(1),
        acc_parts.stride(2),
        acc_parts.stride(3),
        out_buf.stride(1),
        out_buf.stride(2),
        num_splits,
        head_dim,
        block_d,
        num_warps=4,
    )
    return out_buf.to(query.dtype).unsqueeze(2)


def _paged_attention_decode_torch(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    _, num_heads, q_len, head_dim = query.shape
    if q_len != 1:
        raise ValueError("paged decode expects q_len=1")
    block_size = k_cache.shape[2]
    num_blocks = math.ceil(seq_len / block_size)

    q = query[0, :, 0, :].float()
    kv_indices = torch.arange(num_heads, device=query.device) // num_kv_groups
    m_i = torch.full((num_heads,), float("-inf"), device=query.device)
    l_i = torch.zeros(num_heads, device=query.device)
    acc = torch.zeros(num_heads, head_dim, device=query.device)

    for block_idx in range(num_blocks):
        physical = int(block_table[block_idx].item())
        block_start = block_idx * block_size
        take = min(block_size, seq_len - block_start)
        if take <= 0:
            continue
        k_block = k_cache[physical, :, :take, :].float()
        v_block = v_cache[physical, :, :take, :].float()
        k_block = k_block[kv_indices]
        v_block = v_block[kv_indices]
        scores = torch.einsum("hd,htd->ht", q, k_block) * scaling
        block_max = scores.max(dim=-1).values
        m_new = torch.maximum(m_i, block_max)
        exp_scores = torch.exp(scores - m_new.unsqueeze(-1))
        alpha = torch.exp(m_i - m_new)
        l_i = l_i * alpha + exp_scores.sum(dim=-1)
        acc = acc * alpha.unsqueeze(-1) + torch.einsum("ht,htd->hd", exp_scores, v_block)
        m_i = m_new

    out = (acc / l_i.unsqueeze(-1)).to(query.dtype)
    return out.unsqueeze(0).unsqueeze(2)


def _paged_attention_decode_batched_torch(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scaling: float,
    num_kv_groups: int,
    *,
    pool: GpuKVBlockPool | None = None,
    layer_idx: int | None = None,
) -> torch.Tensor:
    """Torch fallback: one batched gather + SDPA (not a Python per-row loop)."""
    if pool is not None and layer_idx is not None:
        keys, values = pool.gather_layer_batch(layer_idx, block_tables, seq_lens)
        return reference_sdpa_attention(query, keys, values, scaling, num_kv_groups)

    # Legacy per-row path when pool handle is unavailable.
    batch_size = query.shape[0]
    outs = []
    max_blocks = block_tables.shape[1]
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        nlog = min(max_blocks, math.ceil(seq_len / k_cache.shape[2]))
        outs.append(
            _paged_attention_decode_torch(
                query[b : b + 1],
                k_cache,
                v_cache,
                block_tables[b, :nlog],
                seq_len,
                scaling,
                num_kv_groups,
            )
        )
    return torch.cat(outs, dim=0)


def paged_attention_decode_batched(
    query: torch.Tensor,
    metadata: PagedDecodeBatchMetadata,
    num_kv_groups: int,
    scaling: float,
) -> torch.Tensor:
    """Batched decode: ``query [B, heads, 1, dim]``."""
    if query.shape[0] != metadata.block_tables.shape[0]:
        raise ValueError("query batch size must match block_tables rows")
    pool = metadata.pool
    pool._ensure_pools()
    k_layer = pool.k_layer_view(metadata.layer_idx)
    v_layer = pool.v_layer_view(metadata.layer_idx)
    block_tables = metadata.block_tables
    seq_lens = metadata.seq_lens.to(device=query.device, dtype=torch.int32)

    if _TRITON_AVAILABLE and query.is_cuda:
        return _paged_attention_decode_batched_triton(
            query, k_layer, v_layer, block_tables, seq_lens, scaling, num_kv_groups
        )
    return _paged_attention_decode_batched_torch(
        query,
        k_layer,
        v_layer,
        block_tables,
        seq_lens,
        scaling,
        num_kv_groups,
        pool=pool,
        layer_idx=metadata.layer_idx,
    )


def paged_attention_decode(
    query: torch.Tensor,
    pool: GpuKVBlockPool,
    layer_idx: int,
    block_ids: list[int],
    seq_len: int,
    num_kv_groups: int,
    scaling: float,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode-step attention for batch=1 (back-compat wrapper)."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive for paged decode")
    num_logical = math.ceil(seq_len / pool.block_size_tokens)
    if block_table is None:
        table = torch.tensor(block_ids[:num_logical], device=query.device, dtype=torch.int32)
    else:
        table = block_table[:num_logical]
    meta = PagedDecodeBatchMetadata(
        pool=pool,
        layer_idx=layer_idx,
        block_tables=table.unsqueeze(0),
        seq_lens=torch.tensor([seq_len], device=query.device, dtype=torch.int32),
    )
    return paged_attention_decode_batched(query, meta, num_kv_groups, scaling)


def reference_sdpa_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    if num_kv_groups > 1:
        keys = keys.repeat_interleave(num_kv_groups, dim=1)
        values = values.repeat_interleave(num_kv_groups, dim=1)
    attn = torch.matmul(query, keys.transpose(2, 3)) * scaling
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(attn, values)
