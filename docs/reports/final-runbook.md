# Benchmark and Chaos Commands

## Full baseline run (external server)
`python bench/harness/run.py --system vllm --base-url http://127.0.0.1:8000/v1/chat/completions --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --vllm-version 0.8.5 --run-id vllm-baseline-final --gpu-type A100 --gpu-hour-usd 3.40 --enable-gpu-sampling`

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
