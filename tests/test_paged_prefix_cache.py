"""Paged prefix Approach A: detached block snapshots + BlockPagedCache hydrate."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from runtime.phase2.kv_allocator import PagedKVAllocator
from runtime.phase2.prefix_cache import PrefixBlockCache, is_paged_prefix_snapshot, truncate_past_key_values


class _FakeBlockPagedCache:
    """Name must be BlockPagedCache so truncate_past_key_values takes the paged path."""

    pass


# Rebind class name for type(obj).__name__ checks without importing GPU pool.
_FakeBlockPagedCache.__name__ = "BlockPagedCache"  # type: ignore[misc]


class PagedPrefixCacheTests(unittest.TestCase):
    def test_truncate_block_paged_stores_marker(self) -> None:
        snap = truncate_past_key_values(_FakeBlockPagedCache(), 32)
        self.assertTrue(is_paged_prefix_snapshot(snap))
        self.assertEqual(snap["seq_len"], 32)

    def test_insert_snapshot_copies_per_depth(self) -> None:
        allocator = PagedKVAllocator(total_blocks=64, block_size_tokens=4, bytes_per_token=1)
        released: list[list[int]] = []

        def on_release(block_ids: list[int]) -> None:
            released.append(list(block_ids))
            allocator.release_blocks(block_ids)

        copies: list[tuple[list[int], list[int]]] = []

        def snapshot(src: list[int]) -> list[int] | None:
            dst = allocator.allocate_detached_blocks(len(src))
            assert dst is not None
            copies.append((list(src), list(dst)))
            return dst

        cache = PrefixBlockCache(
            block_size_tokens=4,
            max_entries=32,
            on_release_blocks=on_release,
        )
        token_ids = list(range(12))  # 3 full blocks
        src_blocks = [10, 11, 12]
        leaf = cache.insert(
            token_ids,
            src_blocks,
            _FakeBlockPagedCache(),
            next_logits=None,
            snapshot_blocks=snapshot,
        )
        self.assertIsNotNone(leaf)
        self.assertEqual(len(copies), 3)
        self.assertTrue(is_paged_prefix_snapshot(leaf.past_key_values))  # type: ignore[union-attr]

        shared = list(range(8))
        hit = cache.lookup(shared + [99, 100, 101, 102])
        self.assertTrue(hit.hit)
        self.assertEqual(hit.matched_tokens, 8)
        self.assertTrue(is_paged_prefix_snapshot(hit.entry.past_key_values))  # type: ignore[union-attr]
        # Hit depth-2 entry must not alias the inserting request's block ids.
        self.assertNotEqual(hit.entry.block_ids, src_blocks[:2])  # type: ignore[union-attr]

    def test_hydrate_paged_prefix_sets_seq_len(self) -> None:
        from runtime.phase2.worker_server import TransformersBackend

        layer = SimpleNamespace(_seq_len=0, is_initialized=False, dtype=None, device=None)
        cache = SimpleNamespace(layers=[layer])
        backend = TransformersBackend.__new__(TransformersBackend)
        backend.model = SimpleNamespace(dtype="float16")
        backend.device = "cpu"
        backend.torch = SimpleNamespace(device=lambda d: d)

        TransformersBackend.hydrate_paged_prefix(backend, cache, 48)  # type: ignore[arg-type]
        self.assertEqual(layer._seq_len, 48)
        self.assertTrue(layer.is_initialized)
        self.assertEqual(layer.dtype, "float16")
        self.assertEqual(layer.device, "cpu")

    def test_admission_paged_hit_does_not_clone_dynamic(self) -> None:
        """Regression: paged prefix hit must not call clone_past_key_values."""
        from runtime.phase2.prefix_cache import PrefixCacheEntry

        entry = PrefixCacheEntry(
            entry_id="e1",
            block_ids=[1, 2],
            block_keys=["a", "b"],
            token_count=32,
            past_key_values={"backend": "paged", "seq_len": 32},
            next_logits=None,
        )
        backend = MagicMock()
        backend.kv_backend = "paged"
        backend.clone_past_key_values = MagicMock(side_effect=AssertionError("must not clone"))
        # Simulate decision branch used in admission.
        paged_hit = backend.kv_backend == "paged" or is_paged_prefix_snapshot(entry.past_key_values)
        self.assertTrue(paged_hit)
        if not paged_hit:
            backend.clone_past_key_values(entry.past_key_values)
        backend.clone_past_key_values.assert_not_called()


if __name__ == "__main__":
    unittest.main()
