# Phase Status

## Phase 1 (functional baseline)
- Implemented: OpenAI-compatible `/v1/chat/completions` server.
- Implemented: request-by-request serving (no batching).
- Artifact: `bench/results/phase1-baseline.csv`
- Artifact: `bench/results/phase1-baseline.manifest.json`

## Phase 2 (functional architecture reset)
- Implemented: split coordinator/worker services.
- Implemented: naive batching in worker queue loop.
- Artifact: `bench/results/phase2-baseline.csv`
- Artifact: `bench/results/phase2-baseline.manifest.json`

## Gaps vs locked stack
- Current Phase 2 services are Python stand-ins for fast iteration.
- Strict locked-stack compliance still requires C++ coordinator and C++/CUDA worker with gRPC hot path.
- Harness TTFT and inter-token latency are measured from streamed assistant token chunks. Non-streaming responses are not used for these timing columns.
- Phase 1 and Phase 2 local runtime benchmarks measure deterministic stub-token generation, not real model inference. Their manifests must use `inference_mode: stub_token_generation`.
- GPU metric columns are valid only when manifest `gpu_metrics_valid` is `true`. If GPU sampling is disabled or `nvidia-smi` yields no samples, GPU CSV columns are zeros and must not be reported as measured utilization or memory.
