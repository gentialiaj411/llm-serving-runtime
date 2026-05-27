from __future__ import annotations

import unittest

import torch

from runtime.phase2.paged_kv_kernel import GpuKVBlockPool


class PagedKVKernelTests(unittest.TestCase):
    def test_write_and_gather_roundtrip(self) -> None:
        pool = GpuKVBlockPool(
            num_blocks=4,
            block_size_tokens=2,
            num_layers=1,
            num_kv_heads=2,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
        )
        block_ids = [0, 1]
        k = torch.ones((1, 2, 2, 4))
        v = torch.ones((1, 2, 2, 4)) * 2
        pool.write_tokens(0, block_ids, 0, k, v)
        keys, values = pool.gather_layer(0, block_ids, 2)
        self.assertEqual(keys.shape, (1, 2, 2, 4))
        self.assertTrue(torch.allclose(keys, k))
        self.assertTrue(torch.allclose(values, v))


if __name__ == "__main__":
    unittest.main()
