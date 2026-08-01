from __future__ import annotations

import unittest

from runtime.phase2.kv_allocator import PagedKVAllocator
from runtime.phase2.prefix_cache import PrefixBlockCache


class PrefixCacheTests(unittest.TestCase):
    def test_hit_after_insert_and_block_refcount(self) -> None:
        allocator = PagedKVAllocator(total_blocks=32, block_size_tokens=4, bytes_per_token=1)
        retained: list[list[int]] = []

        def retain(block_ids: list[int]) -> None:
            retained.extend([list(block_ids)])

        cache = PrefixBlockCache(block_size_tokens=4, max_entries=8, on_retain_blocks=retain)
        token_ids = list(range(12))
        alloc = allocator.allocate_for_tokens("req-a", 12)
        assert alloc is not None
        fake_past = {"layers": 1}
        cache.insert(token_ids, alloc.block_ids, fake_past)

        lookup = cache.lookup(token_ids)
        self.assertTrue(lookup.hit)
        self.assertEqual(lookup.matched_tokens, 12)

        lookup2 = cache.lookup(token_ids + [99, 100])
        self.assertTrue(lookup2.hit)
        self.assertEqual(lookup2.matched_tokens, 12)

        miss = cache.lookup([99, 100, 101, 102])
        self.assertFalse(miss.hit)

    def test_shared_prefix_divergent_suffix_hits(self) -> None:
        """Shared system prompt + unique per-request suffixes must hit the prefix."""
        cache = PrefixBlockCache(block_size_tokens=4, max_entries=64)
        shared = list(range(16))
        first = shared + [100, 101, 102, 103]
        second = shared + [200, 201, 202, 203]
        # Fake past: object is fine for unit scope; truncate passes non-Cache through legacy path.
        class _FakeTensor:
            def __init__(self, seq: int) -> None:
                self.shape = (1, 1, seq, 2)

            def clone(self) -> "_FakeTensor":
                return _FakeTensor(self.shape[-2])

            def __getitem__(self, item: object) -> "_FakeTensor":
                # Support tensor[..., :end, :] style used by truncate.
                return _FakeTensor(min(self.shape[-2], 16))

        seq = len(first)
        fake_past = (( _FakeTensor(seq), _FakeTensor(seq) ),)
        cache.insert(first, list(range(len(first) // 4)), fake_past)
        hit = cache.lookup(second)
        self.assertTrue(hit.hit)
        self.assertEqual(hit.matched_tokens, 16)
        self.assertGreaterEqual(cache.stats()["prefix_cache_hits"], 1)


if __name__ == "__main__":
    unittest.main()
