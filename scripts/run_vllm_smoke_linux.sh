#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-vllm-smoke}"
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${PORT:-8093}"
BASE_URL="http://${VLLM_HOST}:${PORT}/v1/chat/completions"
VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
GPU_TYPE="${GPU_TYPE:-unknown}"
GPU_HOUR_USD="${GPU_HOUR_USD:-0.80}"
SCENARIOS="${SCENARIOS:-bench/scenarios/smoke.yaml}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  CUDA_HOME="$(python - <<'PY'
import pathlib
import sysconfig

site = pathlib.Path(sysconfig.get_paths()["purelib"])
for candidate in site.glob("nvidia/cu*/bin/nvcc"):
    print(candidate.parent.parent)
    raise SystemExit(0)
PY
)"
fi
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

LOG_FILE="/tmp/${RUN_ID}.vllm.log"
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${PORT}" \
  --dtype float16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  >"${LOG_FILE}" 2>&1 &
VLLM_PID=$!

cleanup() {
  kill "${VLLM_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 240); do
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "vLLM exited early. Log: ${LOG_FILE}"
    tail -n 160 "${LOG_FILE}" || true
    exit 1
  fi
  if curl -fsS "http://${VLLM_HOST}:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://${VLLM_HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "vLLM did not become healthy. Log: ${LOG_FILE}"
  tail -n 160 "${LOG_FILE}" || true
  exit 1
fi

python bench/harness/run.py \
  --system vllm \
  --inference-mode real_model_inference \
  --determinism-check warn \
  --scenarios "${SCENARIOS}" \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --vllm-version "${VLLM_VERSION}" \
  --run-id "${RUN_ID}" \
  --gpu-type "${GPU_TYPE}" \
  --gpu-hour-usd "${GPU_HOUR_USD}" \
  --enable-gpu-sampling \
  --warmup-requests 1
