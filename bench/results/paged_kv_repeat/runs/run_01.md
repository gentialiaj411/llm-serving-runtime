# Paged KV — measured GPU memory (Qwen2-1.5B)

- Run UTC: `2026-06-16T02:26:29.862713+00:00`
- Model: `Qwen/Qwen2-1.5B-Instruct`
- GPU: `NVIDIA GeForce RTX 5070 Laptop GPU`
- Workload: `32` requests, variable prompt/output (seed `5070`), max_active `8`

## Peak memory

| Path | torch.cuda.max_memory_allocated (MB) | nvidia-smi peak used (MB) | tokens/sec |
|------|--------------------------------------|---------------------------|------------|
| Contiguous (`PHASE2_KV_BACKEND=reserved`) | 3373.6 | 4997.0 | 21.42 |
| Paged kernel (`PHASE2_KV_BACKEND=paged`) | 4224.2 | 5993.0 | 3.20 |

- Paged peak memory (PyTorch peak): **+25.21% higher** vs contiguous (`paged_vs_contiguous_torch_peak_delta_percent`; positive = paged uses more memory)
- Paged peak memory (nvidia-smi peak): **+19.93% higher** vs contiguous (`paged_vs_contiguous_nvidia_smi_peak_delta_percent`; positive = paged uses more memory)
## Which memory metric to headline

**Headline for resume / external claims:** logical KV efficiency and concurrency
under fixed block budget (`bench/results/paged_kv_concurrency_scale.json`).
Use **`paged_kv_pool_peak_bytes`** from worker `/metrics` for KV-specific bytes.

PyTorch `max_memory_allocated()` and nvidia-smi can diverge; paged is not always
lower on both at moderate concurrency:

- **Contiguous (`reserved`)** admits each request with a worst-case `contiguous_kv_hold`
  tensor (`[layers, 2, kv_heads, token_capacity, head_dim]`) *plus* Hugging Face
  `StaticCache` decode storage. Driver-resident footprint can spike at high
  `max_active × token_capacity`.
- **Paged (`paged`)** stores KV in a shared `GpuKVBlockPool`; only touched physical
  blocks are populated. Logical block usage (`peak_kv_bytes` from `PagedKVAllocator`)
  matches contiguous on this workload (~806 MB); **`paged_kv_pool_peak_bytes`** tracks
  actual pool block bytes (~134 MB in repeat runs).
- **Why PyTorch peak is higher on paged:** the pool uses a growable slab
  (`_grow_pools` doubles `[num_blocks, layers, …]` capacity).
  `torch.cuda.max_memory_allocated()` counts the expanded slab plus fp32 Triton
  scratch (`m_parts`/`l_parts`/`acc_parts` in `paged_attention_triton.py`).
- **Why nvidia-smi can be higher on paged at max_active=8:** the pool slab is
  driver-resident even when logical fill is low; repeat runs (5×) show paged
  nvidia-smi median ~5444 MB vs contiguous ~4797 MB on this workload.

Throughput: see `bench/results/paged_kv_repeat.json` (contiguous median 19.91 tok/s; paged median 3.20 tok/s — check per-run `success_rate` before comparing paged).
## Notes

- Contiguous baseline uses explicit worst-case GPU KV reservation (`PHASE2_KV_BACKEND=reserved`) plus `StaticCache` for decode.
- Paged path uses `GpuKVBlockPool` + `BlockPagedCache` with `PagedKVAllocator` block admission/free.
- Supersedes modeled-only claim in `bench/results/kv-pressure.json` (see `CLAIMS_MATRIX.md`).

Raw traces: `bench/results/paged_kv_real_gpu.json`, `bench/results/paged_kv_real_gpu_smi.csv`.
