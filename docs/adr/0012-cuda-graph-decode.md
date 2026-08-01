# ADR-0012: CUDA Graph Decode

- Status: Accepted (partial — HF capture blocked)
- Date: 2026-07-31

## Context

Phase 2 showed paged continuous batching can keep launch growth near-flat (≈1.33× from B=1→8 under WSL+Triton), but each decode step still pays full CPU launch overhead for the transformer forward. Production engines capture the steady-state decode step into a CUDA graph so replay skips CPU launch work.

Phase 2’s persistent batch slots provide the stable addresses and static batch shapes CUDA graphs require.

## Decision

Add optional CUDA-graph support gated by `PHASE2_CUDA_GRAPH=1`:

- Manager + bucketed `(batch_size, max_blocks)` capture/replay path in `runtime/phase2/cuda_graph_decode.py`.
- Graph-safe in-place KV seq updates (`_graph_safe`) and static `input_ids` / `cache_position` buffers.
- Eager fallback when no graph is captured.
- **HF Qwen2+paged capture is opt-in** via `PHASE2_CUDA_GRAPH_TRY_HF=1`. Default is off because capture currently fails with `cudaErrorStreamCaptureInvalidated` and can sticky-error the CUDA context.
- A tiny `Linear` smoke (`capture_replay_linear_smoke`) proves `torch.cuda.CUDAGraph` itself works on this GPU.

## Consequences

- With default flags, `PHASE2_CUDA_GRAPH=1` installs the manager but decode stays eager until HF capture is fixed or `TRY_HF=1` succeeds.
- Do **not** claim a tok/s win from CUDA graphs on the HF paged path until `hf_paged_graph_captured=true` in `bench/results/cuda_graph_comparison.json`.
- Speculative decoding / LoRA hot-swap are out of scope for the first graph path.

## References

- ADR-0005 Paged attention kernel
- `runtime/phase2/cuda_graph_decode.py`
- `bench/results/cuda_graph_comparison.json`
- `tests/test_cuda_graph_decode.py`
