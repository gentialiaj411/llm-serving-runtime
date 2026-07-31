# Prefix KV cache benchmark

- Model: `Qwen/Qwen2-1.5B-Instruct`
- Shared prefix tokens: `512`
- Requests: `20`, concurrency `4`, decode `32` tokens

| Mode | tokens/sec | success | hit rate | evictions |
|------|------------|---------|----------|-----------|
| Baseline (cache off) | 10.76 | 1.00 | n/a | n/a |
| Prefix cache on | 15.81 | 1.00 | 0.95 | 0 |

- Throughput improvement: **46.97%**

Artifact: `bench/results/prefix_cache.json`
