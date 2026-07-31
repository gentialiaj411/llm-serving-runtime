#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv-vllm-bench/bin/python}"
RUN_ID="${RUN_ID:-orcaforge-head-to-head-tonight}"
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
SCENARIOS="${SCENARIOS:-bench/scenarios/raw_throughput_tonight.yaml}"
GPU_TYPE="${GPU_TYPE:-rtx5070-laptop}"
GPU_HOUR_USD="${GPU_HOUR_USD:-2.50}"
PREFIX_CACHE="${PREFIX_CACHE:-0}"

export PHASE2_BACKEND="${PHASE2_BACKEND:-transformers}"
export HF_MODEL_ID="${MODEL}"
export HF_TORCH_DTYPE="${HF_TORCH_DTYPE:-float16}"
export HF_DEVICE="${HF_DEVICE:-cuda}"
export PHASE2_PREFIX_CACHE="${PREFIX_CACHE}"
export PHASE2_BATCH_DECODE_STEPS="${PHASE2_BATCH_DECODE_STEPS:-1}"
export PHASE2_DECODE_STEP_MS="${PHASE2_DECODE_STEP_MS:-1}"
export KV_TOTAL_BLOCKS="${KV_TOTAL_BLOCKS:-4096}"
export KV_BLOCK_SIZE_TOKENS="${KV_BLOCK_SIZE_TOKENS:-16}"

WORKER_PORT="${WORKER_PORT:-8102}"
COORDINATOR_PORT="${COORDINATOR_PORT:-8101}"
WORKER_LOG="/tmp/${RUN_ID}.worker.log"
COORDINATOR_LOG="/tmp/${RUN_ID}.coordinator.log"

"${PYTHON_BIN}" -m uvicorn runtime.phase2.worker_server:app \
  --host 127.0.0.1 \
  --port "${WORKER_PORT}" \
  --log-level warning \
  >"${WORKER_LOG}" 2>&1 &
WORKER_PID=$!

cleanup() {
  kill "${WORKER_PID}" "${COORDINATOR_PID:-0}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 360); do
  if curl -fsS "http://127.0.0.1:${WORKER_PORT}/healthz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${WORKER_PID}" >/dev/null 2>&1; then
    echo "worker exited; log=${WORKER_LOG}"
    tail -n 120 "${WORKER_LOG}" || true
    exit 1
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${WORKER_PORT}/healthz" >/dev/null

export WORKER_URLS="http://127.0.0.1:${WORKER_PORT}"
export COORDINATOR_HEALTH_UNHEALTHY_THRESHOLD="${COORDINATOR_HEALTH_UNHEALTHY_THRESHOLD:-5}"
export COORDINATOR_DEFAULT_DEADLINE_MS="${COORDINATOR_DEFAULT_DEADLINE_MS:-900000}"
export COORDINATOR_ADMISSION_MAX="${COORDINATOR_ADMISSION_MAX:-128}"
export COORDINATOR_ADMISSION_MIN="${COORDINATOR_ADMISSION_MIN:-128}"

"${PYTHON_BIN}" -m uvicorn runtime.phase2.coordinator_server:app \
  --host 127.0.0.1 \
  --port "${COORDINATOR_PORT}" \
  --log-level warning \
  >"${COORDINATOR_LOG}" 2>&1 &
COORDINATOR_PID=$!

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${COORDINATOR_PORT}/healthz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${COORDINATOR_PID}" >/dev/null 2>&1; then
    echo "coordinator exited; log=${COORDINATOR_LOG}"
    tail -n 120 "${COORDINATOR_LOG}" || true
    exit 1
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${COORDINATOR_PORT}/healthz" >/dev/null

"${PYTHON_BIN}" bench/harness/run.py \
  --system phase2 \
  --base-url "http://127.0.0.1:${COORDINATOR_PORT}/v1/chat/completions" \
  --model "${MODEL}" \
  --vllm-version 0.21.0 \
  --scenarios "${SCENARIOS}" \
  --run-id "${RUN_ID}" \
  --gpu-type "${GPU_TYPE}" \
  --gpu-hour-usd "${GPU_HOUR_USD}" \
  --inference-mode real_model_inference \
  --determinism-check strict \
  --enable-gpu-sampling \
  --warmup-requests 1
