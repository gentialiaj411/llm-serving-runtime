"""Minimal block-paged KV storage and gather kernel for GPU memory measurement.

See docs/adr/0005-paged-attention-kernel.md.
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
        self._k_blocks: dict[int, torch.Tensor] = {}
        self._v_blocks: dict[int, torch.Tensor] = {}
        self._peak_allocated_blocks = 0

    @property
    def block_size_tokens(self) -> int:
        return self.shape.block_size_tokens

    def allocated_block_count(self) -> int:
        return len(self._k_blocks)

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

    def _ensure_block(self, physical: int) -> None:
        if physical in self._k_blocks:
            return
        if physical < 0 or physical >= self.shape.num_blocks:
            raise IndexError(f"physical block id out of range: {physical}")
        self._k_blocks[physical] = torch.zeros(self._block_shape, dtype=self.dtype, device=self.device)
        self._v_blocks[physical] = torch.zeros(self._block_shape, dtype=self.dtype, device=self.device)
        self._peak_allocated_blocks = max(self._peak_allocated_blocks, len(self._k_blocks))

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
        block_size = self.block_size_tokens
        for t in range(kv_len):
            pos = start_pos + t
            logical_block = pos // block_size
            offset = pos % block_size
            if logical_block >= len(block_ids):
                raise IndexError(
                    f"block table underrun at pos={pos}: need block {logical_block}, have {len(block_ids)}"
                )
            physical = block_ids[logical_block]
            self._ensure_block(physical)
            self._k_blocks[physical][layer_idx, :, offset, :] = key_states[:, :, t, :]
            self._v_blocks[physical][layer_idx, :, offset, :] = value_states[:, :, t, :]

    def gather_layer(
        self,
        layer_idx: int,
        block_ids: list[int],
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for attention as [batch=1, heads, seq_len, head_dim]."""
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        block_size = self.block_size_tokens
        num_logical_blocks = math.ceil(seq_len / block_size)
        if num_logical_blocks > len(block_ids):
            raise IndexError("block table underrun during gather")

        pieces_k: list[torch.Tensor] = []
        pieces_v: list[torch.Tensor] = []
        remaining = seq_len
        for logical_idx in range(num_logical_blocks):
            physical = block_ids[logical_idx]
            self._ensure_block(physical)
            take = min(block_size, remaining)
            pieces_k.append(self._k_blocks[physical][layer_idx, :, :take, :].unsqueeze(0))
            pieces_v.append(self._v_blocks[physical][layer_idx, :, :take, :].unsqueeze(0))
            remaining -= take

        keys = torch.cat(pieces_k, dim=2)
        values = torch.cat(pieces_v, dim=2)
        return keys, values

    def reset(self) -> None:
        self._k_blocks.clear()
        self._v_blocks.clear()
        self._peak_allocated_blocks = 0
