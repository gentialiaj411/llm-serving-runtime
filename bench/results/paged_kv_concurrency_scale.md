# Paged KV — concurrency scaling (long sequences)

- Model: `Qwen/Qwen2-1.5B-Instruct` on `NVIDIA GeForce RTX 5070 Laptop GPU`
- Workload: 24 requests, 512+256 tokens each
- KV block budget: `1176` blocks × 16 tokens
- max_active sweep: `[4, 8, 12, 16, 20, 24]`

| max_active | cont success | cont smi peak MB | paged success | paged smi peak MB | cont alloc fails | paged alloc fails |
|------------|--------------|------------------|---------------|-------------------|------------------|-------------------|
| 4 | 1.00 | 5111 | 0.67 | 7570 | 0 | 0 |
| 8 | 1.00 | 5047 | 0.71 | 7827 | 0 | 0 |
| 12 | 1.00 | 5246 | 0.50 | 7836 | 506 | 256 |
| 16 | 1.00 | 5227 | 0.50 | 7836 | 507 | 256 |
| 20 | 1.00 | 7863 | 1.00 | 7777 | 509 | 510 |
| 24 | 1.00 | 5139 | 1.00 | 7790 | 512 | 512 |

- Max sustainable max_active (success=1.0): contiguous **24**, paged **24**

See `docs/adr/0005-paged-attention-kernel.md` for which memory metric to headline.
