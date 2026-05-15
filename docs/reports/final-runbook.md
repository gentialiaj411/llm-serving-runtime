# Benchmark and Chaos Commands

## Full vLLM baseline run (Ubuntu, one command)
`bash scripts/run_vllm_baseline_ubuntu.sh`

Optional overrides:
`RUN_ID=vllm-baseline-final GPU_TYPE=A100 GPU_HOUR_USD=3.40 MAX_MODEL_LEN=1024 GPU_MEM_UTIL=0.60 bash scripts/run_vllm_baseline_ubuntu.sh`

The script pins and records:
- `vllm==0.8.5` in install/run path
- compatible tokenizer stack for this repo harness (`transformers==4.51.3`, `tokenizers==0.21.1`)
- benchmark outputs in `bench/results/<run_id>.csv` and `.manifest.json`

## Full runtime run
`python bench/harness/run.py --system runtime --base-url http://127.0.0.1:8000/v1/chat/completions --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --run-id runtime-final --gpu-type A100 --gpu-hour-usd 3.40 --enable-gpu-sampling`

## Chaos test
1. launch coordinator + N workers
2. collect worker pids
3. run:
`python bench/chaos/run_chaos.py --run-id chaos-final --worker-pids 1234,5678 --duration-s 120 --concurrency 32 --output bench/results/chaos-final.json`

## Completion criteria artifacts
- `bench/results/*baseline*.csv`
- `bench/results/*runtime*.csv`
- `bench/results/*manifest.json`
- `bench/results/chaos-final.json`
