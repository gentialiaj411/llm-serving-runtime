# GPU Compatibility Note

## Local host limitation
- Local GPU: `RTX 5070 Laptop GPU` (compute capability `sm_120`)
- Current `vllm==0.8.5` + available PyTorch binary path in this environment does not provide compatible kernel images for `sm_120`.
- Result: vLLM engine init fails with `RuntimeError: CUDA error: no kernel image is available for execution on the device`.

## Required for true baseline completion
Run baseline command on a supported GPU host (A100/H100 or compatible architecture with working vLLM/PyTorch stack):

```bash
python bench/harness/run.py \
  --system vllm \
  --base-url http://127.0.0.1:8000/v1/chat/completions \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --vllm-version 0.8.5 \
  --run-id vllm-baseline-final \
  --gpu-type A100 \
  --gpu-hour-usd 3.40 \
  --enable-gpu-sampling
```

Then copy `bench/results/vllm-baseline-final.csv` and `.manifest.json` back into this repo.
