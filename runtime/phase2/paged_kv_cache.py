"""Transformers-compatible block-paged KV cache backed by GpuKVBlockPool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.configuration_utils import PreTrainedConfig

from runtime.phase2.paged_kv_kernel import GpuKVBlockPool

if TYPE_CHECKING:
    pass


class BlockPagedLayer(CacheLayerMixin):
    """Single-layer cache backed by shared GPU block pool."""

    is_compileable = False
    is_sliding = False

    def __init__(self, pool: GpuKVBlockPool, layer_idx: int) -> None:
        super().__init__()
        self.pool = pool
        self.layer_idx = layer_idx
        self._block_ids: list[int] = []
        self._seq_len = 0

    def set_block_ids(self, block_ids: list[int]) -> None:
        self._block_ids = list(block_ids)

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if not self._block_ids:
            raise RuntimeError("BlockPagedLayer.update called before block_ids were assigned")

        kv_len = int(key_states.shape[-2])
        self.pool.write_tokens(
            self.layer_idx,
            self._block_ids,
            self._seq_len,
            key_states,
            value_states,
        )
        self._seq_len += kv_len
        return self.pool.gather_layer(self.layer_idx, self._block_ids, self._seq_len)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self._seq_len + query_length, 0

    def get_seq_length(self) -> int:
        return self._seq_len

    def get_max_cache_shape(self) -> int:
        return len(self._block_ids) * self.pool.block_size_tokens

    def reset(self) -> None:
        self._seq_len = 0
        self._block_ids = []
        self.is_initialized = False
        self.keys = None
        self.values = None


class BlockPagedCache(Cache):
    """Multi-layer block-paged cache wired to a shared GpuKVBlockPool."""

    def __init__(self, config: PreTrainedConfig, pool: GpuKVBlockPool) -> None:
        config = config.get_text_config(decoder=True)
        n_layers = int(config.num_hidden_layers)
        if hasattr(config, "num_kv_shared_layers"):
            n_layers -= int(config.num_kv_shared_layers)
        layers = [BlockPagedLayer(pool=pool, layer_idx=i) for i in range(n_layers)]
        super().__init__(layers=layers)
        self.pool = pool

    def set_block_ids(self, block_ids: list[int]) -> None:
        for layer in self.layers:
            assert isinstance(layer, BlockPagedLayer)
            layer.set_block_ids(block_ids)

    def sync_block_ids(self, block_ids: list[int]) -> None:
        """Refresh block table after allocator growth."""
        self.set_block_ids(block_ids)
