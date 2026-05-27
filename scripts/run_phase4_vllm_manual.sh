#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/bhask/Documents/PROJECTS/orcaforge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate awq-env
RUN_ID=vllm-baseline-rtx5070-phase4
MODEL=Qwen/Qwen2-1.5B-Instruct
PORT=8000
HOST=127.0.0.1
LOG=/tmp/${RUN_ID}.vllm.log
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype float16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.60 \
  >"$LOG" 2>&1 &
VPID=$!
cleanup(){ kill "$VPID" >/dev/null 2>&1 || true; }
trap cleanup EXIT
for i in $(seq 1 240); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "http://$HOST:$PORT/health" >/dev/null
python bench/harness/run.py \
  --system vllm \
  --base-url "http://$HOST:$PORT/v1/chat/completions" \
  --model "$MODEL" \
  --vllm-version 0.21.0 \
  --scenarios bench/results/phase2-rtx5070-throughput.scenario.yaml \
  --run-id "$RUN_ID" \
  --gpu-type "RTX 5070 Laptop GPU" \
  --gpu-hour-usd 0.80 \
  --inference-mode real_model_inference \
  --determinism-check warn \
  --enable-gpu-sampling \
  --warmup-requests 1