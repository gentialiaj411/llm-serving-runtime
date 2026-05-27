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
        entry.ref_count += 1
        entry.last_used = time.time()
        self._entries.move_to_end(entry.entry_id)
        if self._on_retain is not None:
            self._on_retain(entry.block_ids)
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
    ) -> PrefixCacheEntry | None:
        blocks = iter_blocks(token_ids, self.block_size_tokens)
        if not blocks or len(block_ids) < len(blocks):
            return None

        keys = [block_hash(chunk) for chunk in blocks]
        entry_id = f"pfx-{keys[0]}-{len(self._entries)}-{time.time_ns()}"
        entry = PrefixCacheEntry(
            entry_id=entry_id,
            block_ids=list(block_ids[: len(blocks)]),
            block_keys=keys,
            token_count=len(blocks) * self.block_size_tokens,
            past_key_values=past_key_values,
            next_logits=next_logits,
            ref_count=0,
        )

        node = self._root
        for idx, key in enumerate(keys):
            if key not in node.children:
                node.children[key] = _TrieNode()
            node = node.children[key]
            if idx == len(keys) - 1:
                node.entry_id = entry_id

        self._entries[entry_id] = entry
        self._entries.move_to_end(entry_id)
        self._inserts += 1
        self._evict_if_needed()
        return entry

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
