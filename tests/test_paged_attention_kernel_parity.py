"""Numerical parity: fused paged decode vs PyTorch SDPA on gathered K/V."""

from __future__ import annotations

import math
import unittest

import torch

from runtime.phase2.paged_attention_triton import (
    paged_attention_decode,
    reference_sdpa_attention,
    triton_available,
)
from runtime.phase2.paged_kv_kernel import GpuKVBlockPool


class PagedAttentionKernelParityTests(unittest.TestCase):
    def _run_parity(
        self,
        *,
        device: str,
        dtype: torch.dtype,
        num_kv_heads: int,
        num_kv_groups: int,
        head_dim: int,
        block_size: int,
        seq_len: int,
    ) -> None:
        num_heads = num_kv_heads * num_kv_groups
        num_blocks = math.ceil(seq_len / block_size) + 1
        pool = GpuKVBlockPool(
            num_blocks=num_blocks,
            block_size_tokens=block_size,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        block_ids = list(range(num_blocks))
        torch.manual_seed(42)
        k = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
        pool.write_tokens(0, block_ids, 0, k, v)

        query = torch.randn(1, num_heads, 1, head_dim, device=device, dtype=dtype)
        scaling = head_dim**-0.5

        keys, values = pool.gather_layer(0, block_ids, seq_len)
        expected = reference_sdpa_attention(query, keys, values, scaling, num_kv_groups)

        if device == "cuda":
            actual = paged_attention_decode(
                query=query,
                pool=pool,
                layer_idx=0,
                block_ids=block_ids,
                seq_len=seq_len,
                num_kv_groups=num_kv_groups,
                scaling=scaling,
            )
        else:
            from runtime.phase2.paged_attention_triton import _paged_attention_decode_torch

            pool._ensure_pools()
            table = torch.tensor(block_ids[: math.ceil(seq_len / block_size)], device=device, dtype=torch.int32)
            actual = _paged_attention_decode_torch(
                query,
                pool.k_layer_view(0),
                pool.v_layer_view(0),
                table,
                seq_len,
                scaling,
                num_kv_groups,
            )

        max_err = (expected - actual).abs().max().item()
        self.assertLess(max_err, 1e-2, f"max abs error {max_err} exceeds tolerance")

    def test_parity_cpu_fp32_short(self) -> None:
        self._run_parity(
            device="cpu",
            dtype=torch.float32,
            num_kv_heads=2,
            num_kv_groups=2,
            head_dim=32,
            block_size=4,
            seq_len=7,
        )

    def test_parity_cpu_fp32_longer(self) -> None:
        self._run_parity(
            device="cpu",
            dtype=torch.float32,
            num_kv_heads=4,
            num_kv_groups=4,
            head_dim=64,
            block_size=16,
            seq_len=100,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_parity_cuda_fp16(self) -> None:
        self._run_parity(
            device="cuda",
            dtype=torch.float16,
            num_kv_heads=2,
            num_kv_groups=4,
            head_dim=64,
            block_size=16,
            seq_len=128,
        )

    @unittest.skipUnless(torch.cuda.is_available() and triton_available(), "Triton + CUDA required")
    def test_parity_cuda_fp16_triton_path(self) -> None:
        self._run_parity(
            device="cuda",
            dtype=torch.float16,
            num_kv_heads=2,
            num_kv_groups=4,
            head_dim=64,
            block_size=16,
            seq_len=128,
        )

    @unittest.skipUnless(torch.cuda.is_available() and triton_available(), "Triton + CUDA required")
    def test_parity_cuda_fp16_batched_ragged(self) -> None:
        """Batched decode (B=4, ragged seq_lens) vs per-row SDPA reference."""
        device = "cuda"
        dtype = torch.float16
        num_kv_heads = 2
        num_kv_groups = 4
        head_dim = 64
        block_size = 16
        batch_size = 4
        seq_lens = [7, 15, 31, 63]
        max_seq = max(seq_lens)
        num_blocks = math.ceil(max_seq / block_size) + 2

        pool = GpuKVBlockPool(
            num_blocks=num_blocks * batch_size,
            block_size_tokens=block_size,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        torch.manual_seed(99)
        scaling = head_dim**-0.5
        num_heads = num_kv_heads * num_kv_groups

        block_tables = torch.zeros(batch_size, num_blocks, device=device, dtype=torch.int32)
        metadata_rows = []
        for b in range(batch_size):
            block_ids = list(range(b * num_blocks, b * num_blocks + num_blocks))
            sl = seq_lens[b]
            nlog = math.ceil(sl / block_size)
            block_tables[b, :nlog] = torch.tensor(block_ids[:nlog], device=device, dtype=torch.int32)
            k = torch.randn(1, num_kv_heads, sl, head_dim, device=device, dtype=dtype)
            v = torch.randn(1, num_kv_heads, sl, head_dim, device=device, dtype=dtype)
            pool.write_tokens(0, block_ids, 0, k, v)
            metadata_rows.append((block_ids, sl))

        query = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)
        from runtime.phase2.paged_attention_triton import PagedDecodeBatchMetadata, paged_attention_decode_batched

        meta = PagedDecodeBatchMetadata(
            pool=pool,
            layer_idx=0,
            block_tables=block_tables,
            seq_lens=torch.tensor(seq_lens, device=device, dtype=torch.int32),
        )
        actual = paged_attention_decode_batched(query, meta, num_kv_groups, scaling)

        for b in range(batch_size):
            block_ids, sl = metadata_rows[b]
            nlog = math.ceil(sl / block_size)
            keys, values = pool.gather_layer(0, block_ids[:nlog], sl)
            expected = reference_sdpa_attention(
                query[b : b + 1], keys, values, scaling, num_kv_groups
            )
            err = (expected - actual[b : b + 1]).abs().max().item()
            self.assertLess(err, 1e-2, f"batch row {b} max abs error {err}")

    @unittest.skipUnless(torch.cuda.is_available() and triton_available(), "Triton + CUDA required")
    def test_parity_cuda_fp16_triton_long_split(self) -> None:
        """Exercises multi-split path (seq_len >= split threshold) and fused combine."""
        self._run_parity(
            device="cuda",
            dtype=torch.float16,
            num_kv_heads=2,
            num_kv_groups=4,
            head_dim=64,
            block_size=16,
            seq_len=768,
        )


if __name__ == "__main__":
    unittest.main()
