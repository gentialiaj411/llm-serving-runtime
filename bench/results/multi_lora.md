# Multi-LoRA benchmark

- Model: `Qwen/Qwen2-1.5B-Instruct`
- Requests: `20` round-robin across `['base', 'adapter_a', 'adapter_b', 'adapter_c']`

| Mode | tokens/sec | success | adapter swaps |
|------|------------|---------|---------------|
| Base only (`PHASE2_LORA=0`) | 20.65 | 1.00 | n/a |
| Multi-LoRA (`PHASE2_LORA=1`) | 15.49 | 1.00 | 0 |

- Throughput delta (multi vs base-only): **−25.0%** (regression, not a win)
- Verified claim: hot-swap of ≥3 adapters with **no base-model reload** between swaps
- `lora_adapter_swaps_total=0` — swap-count instrumentation **unproven** under mixed load (see `CLAIMS_MATRIX.md` row 13)

Note: synthetic PEFT adapters unless `LORA_ADAPTER_PATHS_JSON` provides real checkpoints.
