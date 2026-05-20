from __future__ import annotations

import unittest

from runtime.phase2.kv_allocator import PagedKVAllocator


class PagedKVAllocatorTests(unittest.TestCase):
    def test_capacity_exhaustion_and_reuse_after_free(self) -> None:
        alloc = PagedKVAllocator(total_blocks=4, block_size_tokens=8)

        a = alloc.allocate_for_tokens("req-a", 8)
        b = alloc.allocate_for_tokens("req-b", 16)
        c = alloc.allocate_for_tokens("req-c", 8)
        d = alloc.allocate_for_tokens("req-d", 8)

        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertIsNotNone(c)
        self.assertIsNone(d)
        self.assertEqual(alloc.stats()["free_blocks"], 0)
        self.assertEqual(alloc.stats()["active_allocations"], 3)

        alloc.free_request("req-b")
        reused = alloc.allocate_for_tokens("req-d", 8)

        self.assertIsNotNone(reused)
        self.assertEqual(reused.request_id, "req-d")
        self.assertEqual(alloc.stats()["free_blocks"], 1)
        self.assertEqual(alloc.stats()["active_allocations"], 3)

    def test_fragmentation_metrics_reflect_free_and_active_mix(self) -> None:
        alloc = PagedKVAllocator(total_blocks=6, block_size_tokens=4)

        alloc.allocate_for_tokens("req-a", 8)
        alloc.allocate_for_tokens("req-b", 4)
        before = alloc.stats()

        self.assertEqual(before["used_blocks"], 3)
        self.assertEqual(before["free_blocks"], 3)
        self.assertEqual(before["active_allocations"], 2)
        self.assertGreater(before["fragmentation_ratio"], 0.0)

        alloc.free_request("req-a")
        after = alloc.stats()

        self.assertEqual(after["used_blocks"], 1)
        self.assertEqual(after["free_blocks"], 5)
        self.assertEqual(after["active_allocations"], 1)
        self.assertLess(after["fragmentation_ratio"], before["fragmentation_ratio"])

        alloc.free_request("req-b")
        final = alloc.stats()

        self.assertEqual(final["free_blocks"], 6)
        self.assertEqual(final["active_allocations"], 0)
        self.assertEqual(final["fragmentation_ratio"], 0.0)

    def test_duplicate_allocation_returns_existing_allocation(self) -> None:
        alloc = PagedKVAllocator(total_blocks=2, block_size_tokens=4)

        first = alloc.allocate_for_tokens("req-a", 6)
        second = alloc.allocate_for_tokens("req-a", 2)

        self.assertIs(first, second)
        self.assertEqual(first.requested_tokens, 6)
        self.assertEqual(alloc.stats()["active_allocations"], 1)


if __name__ == "__main__":
    unittest.main()
