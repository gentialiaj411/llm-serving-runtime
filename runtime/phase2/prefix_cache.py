"""Block-level prefix KV cache with radix hash tree, ref counting, and LRU eviction."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable


def block_hash(token_ids: list[int]) -> str:
    payload = ",".join(str(t) for t in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def iter_blocks(token_ids: list[int], block_size: int) -> list[list[int]]:
    if not token_ids:
        return []
    blocks: list[list[int]] = []
    for start in range(0, len(token_ids), block_size):
        blocks.append(token_ids[start : start + block_size])
    return blocks


def is_paged_prefix_snapshot(past_key_values: Any) -> bool:
    return isinstance(past_key_values, dict) and past_key_values.get("backend") == "paged"


def truncate_past_key_values(past_key_values: Any, n_tokens: int) -> Any:
    """Return a KV snapshot limited to the first n_tokens (clone; never mutate input)."""
    if past_key_values is None or n_tokens <= 0:
        return None
    # BlockPagedCache must not be gathered into DynamicCache — store a marker instead.
    if type(past_key_values).__name__ == "BlockPagedCache" or is_paged_prefix_snapshot(past_key_values):
        return {"backend": "paged", "seq_len": int(n_tokens)}
    try:
        from transformers.cache_utils import Cache, DynamicCache
    except Exception:
        DynamicCache = None  # type: ignore[misc, assignment]
        Cache = None  # type: ignore[misc, assignment]

    if Cache is not None and isinstance(past_key_values, Cache):
        assert DynamicCache is not None
        config = getattr(past_key_values, "config", None)
        new_cache = DynamicCache(config=config) if config is not None else DynamicCache()
        for layer_idx, layer in enumerate(past_key_values.layers):
            if not getattr(layer, "is_initialized", False):
                continue
            keys = layer.keys
            values = layer.values
            if keys is None or values is None or keys.numel() == 0:
                continue
            seq = int(keys.shape[-2])
            end = min(n_tokens, seq)
            new_cache.update(keys[..., :end, :].clone(), values[..., :end, :].clone(), layer_idx)
        if hasattr(new_cache, "crop"):
            new_cache.crop(n_tokens)
        return new_cache

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
    else:
        legacy = past_key_values
    # Opaque unit-test / non-tensor handles: keep as-is.
    if not isinstance(legacy, (tuple, list)):
        return past_key_values
    truncated = []
    for layer in legacy:
        if not isinstance(layer, (tuple, list)):
            return past_key_values
        layer_out = []
        for tensor in layer:
            if tensor is None:
                layer_out.append(None)
            else:
                seq = int(tensor.shape[-2])
                end = min(n_tokens, seq)
                layer_out.append(tensor[..., :end, :].clone())
        truncated.append(tuple(layer_out))
    return tuple(truncated)


@dataclass
class PrefixCacheEntry:
    entry_id: str
    block_ids: list[int]
    block_keys: list[str]
    token_count: int
    past_key_values: Any
    next_logits: Any = None
    ref_count: int = 0
    last_used: float = field(default_factory=time.time)


@dataclass
class PrefixLookupResult:
    matched_tokens: int
    matched_blocks: int
    entry: PrefixCacheEntry | None
    hit: bool


class _TrieNode:
    __slots__ = ("children", "entry_id")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.entry_id: str | None = None


class PrefixBlockCache:
    """Radix tree of block hashes -> shared KV prefix entries."""

    def __init__(
        self,
        *,
        block_size_tokens: int,
        max_entries: int = 256,
        on_retain_blocks: Callable[[list[int]], None] | None = None,
        on_release_blocks: Callable[[list[int]], None] | None = None,
    ) -> None:
        self.block_size_tokens = block_size_tokens
        self.max_entries = max(1, max_entries)
        self._on_retain = on_retain_blocks
        self._on_release = on_release_blocks
        self._root = _TrieNode()
        self._entries: OrderedDict[str, PrefixCacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._inserts = 0

    def stats(self) -> dict[str, float | int]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return {
            "prefix_cache_entries": len(self._entries),
            "prefix_cache_hits": self._hits,
            "prefix_cache_misses": self._misses,
            "prefix_cache_hit_rate": hit_rate,
            "prefix_cache_evictions": self._evictions,
            "prefix_cache_inserts": self._inserts,
        }

    def lookup(self, token_ids: list[int]) -> PrefixLookupResult:
        blocks = iter_blocks(token_ids, self.block_size_tokens)
        node = self._root
        matched_blocks = 0
        last_entry: PrefixCacheEntry | None = None
        for chunk in blocks:
            key = block_hash(chunk)
            child = node.children.get(key)
            if child is None:
                break
            node = child
            matched_blocks += 1
            if node.entry_id is not None:
                last_entry = self._entries.get(node.entry_id)

        if last_entry is None or matched_blocks == 0:
            self._misses += 1
            return PrefixLookupResult(0, 0, None, False)

        matched_tokens = min(len(token_ids), matched_blocks * self.block_size_tokens)
        # Require at least one full block overlap to count as a hit.
        if matched_tokens < self.block_size_tokens:
            self._misses += 1
            return PrefixLookupResult(0, 0, None, False)

        self._hits += 1
        last_entry.last_used = time.time()
        self._entries.move_to_end(last_entry.entry_id)
        return PrefixLookupResult(matched_tokens, matched_blocks, last_entry, True)

    def retain_entry(self, entry: PrefixCacheEntry) -> list[int]:
        """Pin entry against LRU eviction while a request is using it.

        Does not touch allocator refs: paged snapshots are held by
        ``allocate_detached_blocks`` until eviction ``on_release``; dynamic hits
        clone tensors and do not need live block pins.
        """
        entry.ref_count += 1
        entry.last_used = time.time()
        self._entries.move_to_end(entry.entry_id)
        return list(entry.block_ids)

    def release_entry(self, entry_id: str) -> None:
        entry = self._entries.get(entry_id)
        if entry is None:
            return
        entry.ref_count = max(0, entry.ref_count - 1)

    def insert(
        self,
        token_ids: list[int],
        block_ids: list[int],
        past_key_values: Any,
        next_logits: Any = None,
        *,
        snapshot_blocks: Callable[[list[int]], list[int] | None] | None = None,
    ) -> PrefixCacheEntry | None:
        """Insert prompt KV into the radix tree.

        Registers an entry at *every* full-block depth so a later request that
        shares only a prefix (different suffix) can still hit. Leaf-only
        registration made shared-prefix workloads report 0 hits.

        When ``snapshot_blocks`` is provided (paged Approach A), each depth gets
        an independent physical copy of the leading blocks so prefix data survives
        after the inserting request frees its allocation.
        """
        blocks = iter_blocks(token_ids, self.block_size_tokens)
        # Only full blocks are cacheable; drop a trailing partial block.
        full_blocks = [b for b in blocks if len(b) == self.block_size_tokens]
        if not full_blocks or len(block_ids) < len(full_blocks):
            return None

        keys = [block_hash(chunk) for chunk in full_blocks]
        node = self._root
        leaf: PrefixCacheEntry | None = None
        for idx, key in enumerate(keys):
            if key not in node.children:
                node.children[key] = _TrieNode()
            node = node.children[key]
            depth = idx + 1
            n_tok = depth * self.block_size_tokens
            is_leaf = idx == len(keys) - 1
            if node.entry_id is not None and not is_leaf:
                # Keep the first cached snapshot for this prefix depth.
                continue
            src_ids = list(block_ids[:depth])
            if snapshot_blocks is not None:
                stored_ids = snapshot_blocks(src_ids)
                if stored_ids is None:
                    return leaf
                stored_past: Any = {"backend": "paged", "seq_len": n_tok}
            else:
                stored_ids = src_ids
                stored_past = truncate_past_key_values(past_key_values, n_tok)
            entry_id = f"pfx-{keys[0]}-{depth}-{len(self._entries)}-{time.time_ns()}"
            entry = PrefixCacheEntry(
                entry_id=entry_id,
                block_ids=stored_ids,
                block_keys=keys[:depth],
                token_count=n_tok,
                past_key_values=stored_past,
                next_logits=next_logits if is_leaf else None,
                ref_count=0,
            )
            if node.entry_id is not None and is_leaf:
                # Replace leaf payload when re-inserting the same full key path.
                old = self._entries.pop(node.entry_id, None)
                if old is not None and self._on_release is not None:
                    self._on_release(old.block_ids)
            node.entry_id = entry_id
            self._entries[entry_id] = entry
            self._entries.move_to_end(entry_id)
            self._inserts += 1
            leaf = entry
        self._evict_if_needed()
        return leaf

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            entry_id, entry = self._entries.popitem(last=False)
            if entry.ref_count > 0:
                self._entries[entry_id] = entry
                self._entries.move_to_end(entry_id, last=False)
                break
            self._remove_entry(entry_id, entry)
            self._evictions += 1

    def _remove_entry(self, entry_id: str, entry: PrefixCacheEntry) -> None:
        if self._on_release is not None:
            self._on_release(entry.block_ids)
        node = self._root
        for key in entry.block_keys:
            child = node.children.get(key)
            if child is None:
                break
            node = child
        if node.entry_id == entry_id:
            node.entry_id = None
        del self._entries[entry_id]
