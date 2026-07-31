"""Block-paged KV storage and gather helpers for GPU memory measurement.

Fused decode attention lives in ``paged_attention_triton.py`` (no gather on decode).
See ``docs/adr/0005-paged-attention-kernel.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PoolShape:
    num_blocks: int
    block_size_tokens: int
    num_layers: int
    num_kv_heads: int
    head_dim: int


class GpuKVBlockPool:
    """Shared GPU block pool with lazy physical block allocation."""

    def __init__(
        self,
        *,
        num_blocks: int,
        block_size_tokens: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> None:
        if num_blocks <= 0 or block_size_tokens <= 0:
            raise ValueError("num_blocks and block_size_tokens must be positive")
        self.shape = PoolShape(
            num_blocks=num_blocks,
            block_size_tokens=block_size_tokens,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.dtype = dtype
        self.device = torch.device(device)
        self._block_shape = (num_layers, num_kv_heads, block_size_tokens, head_dim)
        self._k_pool: torch.Tensor | None = None
        self._v_pool: torch.Tensor | None = None
        self._used_blocks: set[int] = set()
        self._peak_allocated_blocks = 0

    @property
    def block_size_tokens(self) -> int:
        return self.shape.block_size_tokens

    def allocated_block_count(self) -> int:
        return len(self._used_blocks)

    def peak_allocated_block_count(self) -> int:
        return self._peak_allocated_blocks

    def bytes_per_block(self) -> int:
        elems = (
            self.shape.num_layers
            * self.shape.num_kv_heads
            * self.shape.block_size_tokens
            * self.shape.head_dim
            * 2
        )
        return elems * torch.empty((), dtype=self.dtype).element_size()

    def bytes_for_blocks(self, block_count: int) -> int:
        return block_count * self.bytes_per_block()

    def peak_allocated_bytes(self) -> int:
        return self.bytes_for_blocks(self._peak_allocated_blocks)

    def _ensure_pools(self) -> None:
        if self._k_pool is not None:
            return
        self._grow_pools(min(1, self.shape.num_blocks))

    _POOL_GROWTH_BLOCKS = 8

    def _grow_pools(self, min_blocks: int) -> None:
        needed = min(self.shape.num_blocks, max(min_blocks, 1))
        if self._k_pool is not None and self._k_pool.shape[0] >= needed:
            return
        if self._k_pool is None:
            target = needed
        else:
            # Conservative linear growth — avoid 2x doubling that inflates
            # torch.cuda.max_memory_allocated() with unused slab capacity.
            target = min(
                self.shape.num_blocks,
                max(needed, self._k_pool.shape[0] + self._POOL_GROWTH_BLOCKS),
            )
        shape = (
            target,
            self.shape.num_layers,
            self.shape.num_kv_heads,
            self.shape.block_size_tokens,
            self.shape.head_dim,
        )
        k_new = torch.zeros(shape, dtype=self.dtype, device=self.device)
        v_new = torch.zeros(shape, dtype=self.dtype, device=self.device)
        if self._k_pool is not None:
            existing = self._k_pool.shape[0]
            k_new[:existing].copy_(self._k_pool)
            v_new[:existing].copy_(self._v_pool)
        self._k_pool = k_new
        self._v_pool = v_new

    def _touch_block(self, physical: int) -> None:
        if physical < 0 or physical >= self.shape.num_blocks:
            raise IndexError(f"physical block id out of range: {physical}")
        self._grow_pools(physical + 1)
        self._used_blocks.add(physical)
        self._peak_allocated_blocks = max(self._peak_allocated_blocks, len(self._used_blocks))

    def k_layer_view(self, layer_idx: int) -> torch.Tensor:
        """Per-layer K blocks: ``[num_blocks, num_kv_heads, block_size, head_dim]``."""
        self._ensure_pools()
        assert self._k_pool is not None
        return self._k_pool[:, layer_idx]

    def v_layer_view(self, layer_idx: int) -> torch.Tensor:
        """Per-layer V blocks: ``[num_blocks, num_kv_heads, block_size, head_dim]``."""
        self._ensure_pools()
        assert self._v_pool is not None
        return self._v_pool[:, layer_idx]

    def write_tokens(
        self,
        layer_idx: int,
        block_ids: list[int],
        start_pos: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Write key/value states for token positions [start_pos, start_pos + kv_len)."""
        kv_len = int(key_states.shape[-2])
        if key_states.dim() == 3:
            key_states = key_states.unsqueeze(0)
            value_states = value_states.unsqueeze(0)
        if key_states.dim() == 4 and key_states.shape[0] == 1:
            self.write_tokens_batch(
                layer_idx,
                torch.tensor([block_ids], device=self.device, dtype=torch.int32),
                torch.tensor([start_pos], device=self.device, dtype=torch.int32),
                key_states,
                value_states,
            )
            return
        block_size = self.block_size_tokens
        self._ensure_pools()
        assert self._k_pool is not None
        assert self._v_pool is not None

        for t in range(kv_len):
            pos = start_pos + t
            logical_block = pos // block_size
            offset = pos % block_size
            if logical_block >= len(block_ids):
                raise IndexError(
                    f"block table underrun at pos={pos}: need block {logical_block}, have {len(block_ids)}"
                )
            physical = block_ids[logical_block]
            self._touch_block(physical)
            self._k_pool[physical, layer_idx, :, offset, :] = key_states[:, :, t, :]
            self._v_pool[physical, layer_idx, :, offset, :] = value_states[:, :, t, :]

    def write_tokens_batch(
        self,
        layer_idx: int,
        block_tables: torch.Tensor,
        start_positions: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Vectorized KV write for batch rows.

        ``block_tables`` [B, max_blocks], ``start_positions`` [B], states [B, kv_heads, kv_len, dim].
        """
        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must match shape")
        batch_size = int(key_states.shape[0])
        kv_len = int(key_states.shape[-2])
        if block_tables.shape[0] != batch_size or start_positions.shape[0] != batch_size:
            raise ValueError("batch dimension mismatch in write_tokens_batch")
        block_size = self.block_size_tokens
        self._ensure_pools()
        assert self._k_pool is not None
        assert self._v_pool is not None

        device = self.device
        block_tables = block_tables.to(device=device, dtype=torch.int32)
        start_positions = start_positions.to(device=device, dtype=torch.int64)
        row_idx = torch.arange(batch_size, device=device)

        for t in range(kv_len):
            pos = start_positions + t
            logical = torch.div(pos, block_size, rounding_mode="floor")
            offset = pos % block_size
            if logical.max().item() >= block_tables.shape[1]:
                raise IndexError("block table underrun during batched write")
            physical = block_tables[row_idx, logical]
            self._touch_blocks(physical)
            self._k_pool[physical, layer_idx, :, offset, :] = key_states[:, :, t, :]
            self._v_pool[physical, layer_idx, :, offset, :] = value_states[:, :, t, :]

    def _touch_blocks(self, physical_ids: torch.Tensor) -> None:
        for physical in physical_ids.unique().tolist():
            self._touch_block(int(physical))

    def gather_layer(
        self,
        layer_idx: int,
        block_ids: list[int],
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for attention as [batch=1, heads, seq_len, head_dim] (prefill / parity)."""
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        block_size = self.block_size_tokens
        num_logical_blocks = math.ceil(seq_len / block_size)
        if num_logical_blocks > len(block_ids):
            raise IndexError("block table underrun during gather")

        self._ensure_pools()
        assert self._k_pool is not None
        assert self._v_pool is not None

        pieces_k: list[torch.Tensor] = []
        pieces_v: list[torch.Tensor] = []
        remaining = seq_len
        for logical_idx in range(num_logical_blocks):
            physical = block_ids[logical_idx]
            self._touch_block(physical)
            take = min(block_size, remaining)
            pieces_k.append(self._k_pool[physical, layer_idx, :, :take, :].unsqueeze(0))
            pieces_v.append(self._v_pool[physical, layer_idx, :, :take, :].unsqueeze(0))
            remaining -= take

        keys = torch.cat(pieces_k, dim=2)
        values = torch.cat(pieces_v, dim=2)
        return keys, values

    def gather_layer_batch(
        self,
        layer_idx: int,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for batched prefill: [B, kv_heads, max_seq, head_dim] padded."""
        batch_size = int(block_tables.shape[0])
        max_seq = int(seq_lens.max().item())
        if max_seq <= 0:
            raise ValueError("seq_len must be positive")
        block_size = self.block_size_tokens
        device = self.device
        num_kv_heads = self.shape.num_kv_heads
        head_dim = self.shape.head_dim
        keys = torch.zeros(batch_size, num_kv_heads, max_seq, head_dim, device=device, dtype=self.dtype)
        values = torch.zeros_like(keys)
        for b in range(batch_size):
            seq_len = int(seq_lens[b].item())
            num_logical = math.ceil(seq_len / block_size)
            row_ids = [int(x) for x in block_tables[b, :num_logical].tolist()]
            k, v = self.gather_layer(layer_idx, row_ids, seq_len)
            keys[b, :, :seq_len, :] = k[0, :, :seq_len, :]
            values[b, :, :seq_len, :] = v[0, :, :seq_len, :]
        return keys, values

    def reset(self) -> None:
        self._k_pool = None
        self._v_pool = None
        self._used_blocks.clear()
        self._peak_allocated_blocks = 0
