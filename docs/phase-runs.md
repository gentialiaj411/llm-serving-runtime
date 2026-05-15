# Running Benchmarks

## Local endpoint (phase1 server)
1. `python -m uvicorn frontend.phase1_server:app --host 127.0.0.1 --port 8000`
2. `python bench/harness/run.py --system phase1 --base-url http://127.0.0.1:8000/v1/chat/completions --run-id phase1-local --gpu-type RTX4090 --gpu-hour-usd 0.80 --enable-gpu-sampling`

## Split endpoint (coordinator + worker)
1. `python -m uvicorn runtime.phase2.worker_server:app --host 127.0.0.1 --port 8102`
2. `python -m uvicorn runtime.phase2.coordinator_server:app --host 127.0.0.1 --port 8000`
3. `python bench/harness/run.py --system phase2 --base-url http://127.0.0.1:8000/v1/chat/completions --run-id phase2-local --gpu-type RTX4090 --gpu-hour-usd 0.80 --enable-gpu-sampling`

## vLLM baseline (same harness, existing server)
`python bench/harness/run.py --system vllm --base-url http://127.0.0.1:8000/v1/chat/completions --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --vllm-version 0.8.5 --run-id vllm-baseline --gpu-type A100 --gpu-hour-usd 3.40 --enable-gpu-sampling`

## vLLM baseline (one-command launcher mode)
`python bench/harness/run.py --system vllm --launch-vllm --base-url http://127.0.0.1:8000/v1/chat/completions --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --vllm-version 0.8.5 --run-id vllm-baseline-auto --gpu-type A100 --gpu-hour-usd 3.40 --enable-gpu-sampling --vllm-max-model-len 4096 --vllm-tensor-parallel-size 1`

## Notes
- Determinism check runs automatically and fails if temperature=0 outputs differ.
- TTFT and inter-token latency require streaming responses and are measured from streamed assistant token chunks.
- GPU sampling is time-based via `nvidia-smi` (default every 0.5s) when `--enable-gpu-sampling` is set.
- Check manifest `inference_mode` before comparing benchmark rows. Local Phase 1/Phase 2 runtimes are `stub_token_generation`; vLLM should be `real_model_inference`.
- Check manifest `gpu_metrics_valid` before reporting GPU utilization or memory metrics.
- `--launch-vllm` requires `vllm` to be installed in the current Python environment.
