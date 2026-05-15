# Running Benchmarks

## Local endpoint (phase1 server)
Install shared/runtime dependencies with `python -m pip install -r requirements.txt` or `python -m pip install -r runtime/requirements.txt`.

1. `python -m uvicorn frontend.phase1_server:app --host 127.0.0.1 --port 8000`
2. `python bench/harness/run.py --system phase1 --base-url http://127.0.0.1:8000/v1/chat/completions --run-id phase1-local --gpu-type RTX4090 --gpu-hour-usd 0.80 --enable-gpu-sampling`

## Local endpoint with real model inference
1. Install a GPU-compatible PyTorch build plus `transformers`, or start from `runtime/requirements-transformers.txt` and adjust the PyTorch wheel for the target CUDA stack.
2. Start Phase 1 with `PHASE1_BACKEND=transformers python -m uvicorn frontend.phase1_server:app --host 127.0.0.1 --port 8000`
3. Run `python bench/harness/run.py --system phase1-transformers --inference-mode real_model_inference --determinism-check warn --scenarios bench/scenarios/smoke.yaml --base-url http://127.0.0.1:8000/v1/chat/completions --run-id phase1-transformers-smoke --gpu-type <GPU> --enable-gpu-sampling`

## Split endpoint (coordinator + worker)
1. `python -m uvicorn runtime.phase2.worker_server:app --host 127.0.0.1 --port 8102`
2. `python -m uvicorn runtime.phase2.coordinator_server:app --host 127.0.0.1 --port 8000`
3. `python bench/harness/run.py --system phase2 --base-url http://127.0.0.1:8000/v1/chat/completions --run-id phase2-local --gpu-type RTX4090 --gpu-hour-usd 0.80 --enable-gpu-sampling`

## vLLM baseline (same harness, existing server)
`python bench/harness/run.py --system vllm --base-url http://127.0.0.1:8000/v1/chat/completions --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --vllm-version 0.21.0 --run-id vllm-baseline --gpu-type A100 --gpu-hour-usd 3.40 --inference-mode real_model_inference --determinism-check warn --enable-gpu-sampling --warmup-requests 1`

## vLLM baseline (one-command launcher mode)
`python bench/harness/run.py --system vllm --launch-vllm --base-url http://127.0.0.1:8000/v1/chat/completions --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --vllm-version 0.21.0 --run-id vllm-baseline-auto --gpu-type A100 --gpu-hour-usd 3.40 --inference-mode real_model_inference --determinism-check warn --enable-gpu-sampling --warmup-requests 1 --vllm-max-model-len 4096 --vllm-tensor-parallel-size 1`

## vLLM smoke on WSL/Linux
`RUN_ID=vllm-smoke-20260515 GPU_TYPE=RTX5070Laptop GPU_HOUR_USD=0.80 bash scripts/run_vllm_smoke_linux.sh`

The smoke runner starts vLLM, waits for `/health`, and then runs the same streaming harness against `bench/scenarios/smoke.yaml`. It defaults `VLLM_USE_FLASHINFER_SAMPLER=0` because the local RTX 5070 Laptop WSL stack needed the PyTorch-native sampler path to avoid FlashInfer JIT/CUDA header mismatches.

## Real-inference artifacts
- `bench/results/phase1-transformers-smoke-20260515.csv`
- `bench/results/phase1-transformers-smoke-20260515.manifest.json`
- `bench/results/vllm-smoke-20260515.csv`
- `bench/results/vllm-smoke-20260515.manifest.json`

Both runs use TinyLlama/TinyLlama-1.1B-Chat-v1.0 on the same RTX 5070 Laptop GPU with `inference_mode: real_model_inference` and `gpu_metrics_valid: true`.

## Notes
- Determinism check runs automatically and fails if temperature=0 outputs differ.
- TTFT and inter-token latency require streaming responses and are measured from streamed assistant token chunks.
- The harness sends one warmup streaming request per scenario/concurrency by default before timed measurement; override with `--warmup-requests`.
- GPU sampling is time-based via `nvidia-smi` (default every 0.5s) when `--enable-gpu-sampling` is set.
- Check manifest `inference_mode` before comparing benchmark rows. Local Phase 1/Phase 2 stub runtimes are `stub_token_generation`; Phase 1 Transformers and vLLM runs should be `real_model_inference`.
- Check manifest `gpu_metrics_valid` before reporting GPU utilization or memory metrics.
- For real model backends, use `--determinism-check warn` or `--determinism-check skip` if temperature-0 output can vary from kernel or batching nondeterminism.
- Full vLLM baselines should still be run on a stable Linux GPU host for larger scenario sets; the committed smoke artifact is intentionally lightweight.
- `--launch-vllm` requires `vllm` to be installed in the current Python environment.
