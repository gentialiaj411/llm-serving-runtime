"""Transformers-compatible block-paged KV cache backed by GpuKVBlockPool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.configuration_utils import PreTrainedConfig

from runtime.phase2.paged_attention_triton import (
    PagedDecodeBatchMetadata,
    reset_paged_decode_batch_metadata,
    set_paged_decode_batch_metadata,
)
from runtime.phase2.paged_kv_kernel import GpuKVBlockPool

if TYPE_CHECKING:
    pass


class BlockPagedLayer(CacheLayerMixin):
    """Single-layer cache backed by shared GPU block pool (batch size 1)."""

    is_compileable = False
    is_sliding = False

    def __init__(self, pool: GpuKVBlockPool, layer_idx: int) -> None:
        super().__init__()
        self.pool = pool
        self.layer_idx = layer_idx
        self._block_ids: list[int] = []
        self._block_table: torch.Tensor | None = None
        self._seq_len = 0

    def set_block_ids(self, block_ids: list[int], block_table: torch.Tensor | None = None) -> None:
        self._block_ids = list(block_ids)
        self._block_table = block_table

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
        start_pos = self._seq_len
        if key_states.dim() == 3:
            key_states = key_states.unsqueeze(0)
            value_states = value_states.unsqueeze(0)
        self.pool.write_tokens_batch(
            self.layer_idx,
            self._block_table.unsqueeze(0) if self._block_table is not None else torch.tensor(
                [self._block_ids], device=self.pool.device, dtype=torch.int32
            ),
            torch.tensor([start_pos], device=self.pool.device, dtype=torch.int32),
            key_states,
            value_states,
        )
        self._seq_len += kv_len

        if kv_len == 1:
            block_table = self._block_table
            if block_table is None:
                block_table = torch.tensor(self._block_ids, device=self.pool.device, dtype=torch.int32)
            meta = PagedDecodeBatchMetadata(
                pool=self.pool,
                layer_idx=self.layer_idx,
                block_tables=block_table.unsqueeze(0),
                seq_lens=torch.tensor([self._seq_len], device=self.pool.device, dtype=torch.int32),
            )
            set_paged_decode_batch_metadata(meta)
            return key_states, value_states

        set_paged_decode_batch_metadata(None)
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


class BlockPagedBatchLayer(CacheLayerMixin):
    """Single layer for batched block-paged decode."""

    is_compileable = False
    is_sliding = False

    def __init__(self, pool: GpuKVBlockPool, layer_idx: int, parent: BlockPagedBatchCache) -> None:
        super().__init__()
        self.pool = pool
        self.layer_idx = layer_idx
        self._parent = parent
        self._row_seq_lens = parent.seq_lens.clone()

    def _row_lengths(self, parent: BlockPagedBatchCache) -> torch.Tensor:
        return self._row_seq_lens

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        parent = self._parent
        kv_len = int(key_states.shape[-2])
        row_seq_lens = self._row_lengths(parent)
        start_positions = row_seq_lens.clone()
        parent.pool.write_tokens_batch(
            self.layer_idx,
            parent.block_tables,
            start_positions,
            key_states,
            value_states,
        )
        self._row_seq_lens = row_seq_lens + kv_len
        if self.layer_idx == 0:
            parent.seq_lens = self._row_seq_lens.clone()

        if kv_len == 1:
            meta = PagedDecodeBatchMetadata(
                pool=self.pool,
                layer_idx=self.layer_idx,
                block_tables=parent.block_tables,
                seq_lens=parent.seq_lens,
            )
            set_paged_decode_batch_metadata(meta)
            return key_states, value_states

        set_paged_decode_batch_metadata(None)
        return parent.pool.gather_layer_batch(self.layer_idx, parent.block_tables, parent.seq_lens)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return int(self._row_seq_lens.max().item()) + query_length, 0

    def get_seq_length(self) -> int:
        return int(self._row_seq_lens.max().item())

    def get_max_cache_shape(self) -> int:
        return self._parent.block_tables.shape[1] * self.pool.block_size_tokens

    def reset(self) -> None:
        self.is_initialized = False
        self._row_seq_lens = self._parent.seq_lens.clone()


class BlockPagedCache(Cache):
    """Multi-layer block-paged cache wired to a shared GpuKVBlockPool (batch=1)."""

    def __init__(self, config: PreTrainedConfig, pool: GpuKVBlockPool) -> None:
        config = config.get_text_config(decoder=True)
        n_layers = int(config.num_hidden_layers)
        if hasattr(config, "num_kv_shared_layers"):
            n_layers -= int(config.num_kv_shared_layers)
        layers = [BlockPagedLayer(pool=pool, layer_idx=i) for i in range(n_layers)]
        super().__init__(layers=layers)
        self.pool = pool
        self._config = config
        self._block_table: torch.Tensor | None = None
        self.block_table_ids: list[int] = []

    def set_block_ids(self, block_ids: list[int]) -> None:
        if self._block_table is not None and self.block_table_ids == list(block_ids):
            return
        table = torch.tensor(block_ids, device=self.pool.device, dtype=torch.int32)
        for layer in self.layers:
            assert isinstance(layer, BlockPagedLayer)
            layer.set_block_ids(block_ids, table)
        self._block_table = table
        self.block_table_ids = list(block_ids)

    def sync_block_ids(self, block_ids: list[int]) -> None:
        self.set_block_ids(block_ids)

    def block_table(self) -> torch.Tensor | None:
        return self._block_table

    def get_seq_length(self) -> int:
        layer0 = self.layers[0]
        assert isinstance(layer0, BlockPagedLayer)
        return layer0.get_seq_length()


class BlockPagedBatchCache(Cache):
    """Batched block-paged cache for continuous-batch decode (B > 1).

    Supports a persistent max-batch allocation with a free-slot list so admissions
    and completions update row metadata in place instead of rebuilding via
    ``from_caches`` (no KV ``torch.cat``).
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        pool: GpuKVBlockPool,
        *,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_ids_rows: list[list[int]],
    ) -> None:
        config = config.get_text_config(decoder=True)
        n_layers = int(config.num_hidden_layers)
        if hasattr(config, "num_kv_shared_layers"):
            n_layers -= int(config.num_kv_shared_layers)
        self.pool = pool
        self._config = config
        self.block_tables = block_tables.to(device=pool.device, dtype=torch.int32)
        self.seq_lens = seq_lens.to(device=pool.device, dtype=torch.int32)
        self.block_ids_rows = [list(row) for row in block_ids_rows]
        self._batch_size = int(self.block_tables.shape[0])
        self._slot_occupied: list[bool] = [True] * self._batch_size
        self._free_slots: list[int] = []
        self._request_to_slot: dict[str, int] = {}
        layers = [BlockPagedBatchLayer(pool=pool, layer_idx=i, parent=self) for i in range(n_layers)]
        super().__init__(layers=layers)

    @classmethod
    def empty(
        cls,
        config: PreTrainedConfig,
        pool: GpuKVBlockPool,
        *,
        max_batch_size: int,
        max_blocks: int,
    ) -> BlockPagedBatchCache:
        if max_batch_size <= 0 or max_blocks <= 0:
            raise ValueError("max_batch_size and max_blocks must be positive")
        device = pool.device
        tables = torch.zeros(max_batch_size, max_blocks, device=device, dtype=torch.int32)
        seq_lens = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
        rows: list[list[int]] = [[] for _ in range(max_batch_size)]
        cache = cls(
            config,
            pool,
            block_tables=tables,
            seq_lens=seq_lens,
            block_ids_rows=rows,
        )
        cache._slot_occupied = [False] * max_batch_size
        cache._free_slots = list(range(max_batch_size - 1, -1, -1))
        cache._request_to_slot = {}
        return cache

    @classmethod
    def from_caches(cls, caches: list[BlockPagedCache], pool: GpuKVBlockPool) -> BlockPagedBatchCache:
        if not caches:
            raise ValueError("from_caches requires at least one cache")
        config = caches[0]._config
        block_ids_rows = []
        seq_lens_list = []
        for cache in caches:
            layer0 = cache.layers[0]
            assert isinstance(layer0, BlockPagedLayer)
            block_ids_rows.append(list(layer0._block_ids))
            seq_lens_list.append(layer0.get_seq_length())
        max_blocks = max(len(row) for row in block_ids_rows)
        device = pool.device
        tables = torch.zeros(len(caches), max_blocks, device=device, dtype=torch.int32)
        for b, ids in enumerate(block_ids_rows):
            if ids:
                tables[b, : len(ids)] = torch.tensor(ids, device=device, dtype=torch.int32)
        seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.int32)
        cache = cls(
            config,
            pool,
            block_tables=tables,
            seq_lens=seq_lens,
            block_ids_rows=block_ids_rows,
        )
        cache._slot_occupied = [True] * len(caches)
        cache._free_slots = []
        return cache

    def allocate_slot(self, request_id: str) -> int:
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            raise RuntimeError("persistent paged batch has no free slots")
        slot = self._free_slots.pop()
        self._slot_occupied[slot] = True
        self._request_to_slot[request_id] = slot
        return slot

    def release_slot(self, request_id: str) -> None:
        slot = self._request_to_slot.pop(request_id, None)
        if slot is None:
            return
        self._slot_occupied[slot] = False
        self.block_ids_rows[slot] = []
        self.seq_lens[slot] = 0
        self.block_tables[slot].zero_()
        self._free_slots.append(slot)

    def bind_cache(self, request_id: str, cache: BlockPagedCache) -> int:
        slot = self.allocate_slot(request_id)
        layer0 = cache.layers[0]
        assert isinstance(layer0, BlockPagedLayer)
        self.sync_block_ids_row(slot, list(layer0._block_ids))
        seq_len = layer0.get_seq_length()
        self.seq_lens[slot] = int(seq_len)
        for layer in self.layers:
            assert isinstance(layer, BlockPagedBatchLayer)
            layer._row_seq_lens[slot] = int(seq_len)
        return slot

    def slot_of(self, request_id: str) -> int | None:
        return self._request_to_slot.get(request_id)

    def dense_active_cache(self, request_ids: list[str]) -> tuple[BlockPagedBatchCache, list[int]]:
        """Build a dense B-row cache view over selected persistent slots (table copy only)."""
        slots: list[int] = []
        for rid in request_ids:
            slot = self._request_to_slot.get(rid)
            if slot is None:
                raise KeyError(f"request {rid} is not bound to a persistent slot")
            slots.append(slot)
        if not slots:
            raise ValueError("dense_active_cache requires at least one request")
        max_blocks = int(self.block_tables.shape[1])
        device = self.block_tables.device
        tables = torch.zeros(len(slots), max_blocks, device=device, dtype=torch.int32)
        seq_list: list[int] = []
        rows: list[list[int]] = []
        for i, slot in enumerate(slots):
            tables[i] = self.block_tables[slot]
            seq_list.append(int(self.seq_lens[slot].item()))
            rows.append(list(self.block_ids_rows[slot]))
        seq_lens = torch.tensor(seq_list, device=device, dtype=torch.int32)
        dense = BlockPagedBatchCache(
            self._config,
            self.pool,
            block_tables=tables,
            seq_lens=seq_lens,
            block_ids_rows=rows,
        )
        return dense, slots

    def write_back_dense(self, dense: BlockPagedBatchCache, slots: list[int]) -> None:
        """Copy dense-row seq_lens / block tables back into persistent slots after decode."""
        for i, slot in enumerate(slots):
            self.seq_lens[slot] = dense.seq_lens[i]
            self.sync_block_ids_row(slot, list(dense.block_ids_rows[i]))
            for layer in self.layers:
                assert isinstance(layer, BlockPagedBatchLayer)
                layer._row_seq_lens[slot] = int(dense.seq_lens[i].item())

    def extract_cache(self, index: int) -> BlockPagedCache:
        cache = BlockPagedCache(self._config, self.pool)
        block_ids = self.block_ids_rows[index]
        cache.set_block_ids(block_ids)
        seq_len = int(self.seq_lens[index].item())
        for layer in cache.layers:
            assert isinstance(layer, BlockPagedLayer)
            layer._seq_len = seq_len
        return cache

    def get_seq_length(self) -> int:
        return int(self.seq_lens.max().item())

    def ensure_max_blocks(self, max_blocks: int) -> None:
        cur = int(self.block_tables.shape[1])
        if max_blocks <= cur:
            return
        grown = torch.zeros(
            self._batch_size,
            max_blocks,
            device=self.block_tables.device,
            dtype=torch.int32,
        )
        grown[:, :cur] = self.block_tables
        self.block_tables = grown

    def sync_block_ids_row(self, index: int, block_ids: list[int]) -> None:
        self.block_ids_rows[index] = list(block_ids)
        max_blocks = self.block_tables.shape[1]
        if len(block_ids) > max_blocks:
            self.ensure_max_blocks(len(block_ids))
            max_blocks = len(block_ids)
        row = torch.zeros(max_blocks, device=self.block_tables.device, dtype=torch.int32)
        if block_ids:
            row[: len(block_ids)] = torch.tensor(block_ids, device=self.block_tables.device, dtype=torch.int32)
        self.block_tables[index] = row
