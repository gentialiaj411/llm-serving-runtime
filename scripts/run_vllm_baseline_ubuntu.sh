#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-vllm-baseline-final}"
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PORT="${PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
BASE_URL="http://${VLLM_HOST}:${PORT}/v1/chat/completions"
VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
VLLM_VERSION_PIN="${VLLM_VERSION}"
FASTAPI_VERSION="${FASTAPI_VERSION:-0.137.1}"
STARLETTE_VERSION="${STARLETTE_VERSION:-1.3.1}"
PROMETHEUS_FASTAPI_INSTRUMENTATOR_VERSION="${PROMETHEUS_FASTAPI_INSTRUMENTATOR_VERSION:-8.0.0}"
GPU_TYPE="${GPU_TYPE:-unknown}"
GPU_HOUR_USD="${GPU_HOUR_USD:-2.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
SCENARIOS="${SCENARIOS:-bench/scenarios/baseline.yaml}"
READINESS_TIMEOUT_SECS="${READINESS_TIMEOUT_SECS:-600}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "python or python3 is required"
    exit 1
  fi
fi

VENV_DIR="${VENV_DIR:-.venv-vllm-bench}"
USE_VENV=1
if [[ ! -d "${VENV_DIR}" ]]; then
  if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
    USE_VENV=0
  fi
fi
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  USE_VENV=0
fi
if [[ -d "${VENV_DIR}" && "${USE_VENV}" -eq 1 ]]; then
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="python"
  python -m pip install -q --upgrade pip
  python -m pip install -q -r bench/harness/requirements.txt
  # Let vLLM resolve compatible transformers/tokenizers for the pinned vLLM version.
  python -m pip install -q "vllm==${VLLM_VERSION}"
  python -m pip install -q \
    "fastapi==${FASTAPI_VERSION}" \
    "starlette==${STARLETTE_VERSION}" \
    "prometheus-fastapi-instrumentator==${PROMETHEUS_FASTAPI_INSTRUMENTATOR_VERSION}"
  python bench/scripts/patch_vllm_metrics_compat.py
else
  "${PYTHON_BIN}" -m pip install -q --user --break-system-packages --upgrade pip
  "${PYTHON_BIN}" -m pip install -q --user --break-system-packages -r bench/harness/requirements.txt
  # Let vLLM resolve compatible transformers/tokenizers for the pinned vLLM version.
  "${PYTHON_BIN}" -m pip install -q --user --break-system-packages "vllm==${VLLM_VERSION}"
  "${PYTHON_BIN}" -m pip install -q --user --break-system-packages \
    "fastapi==${FASTAPI_VERSION}" \
    "starlette==${STARLETTE_VERSION}" \
    "prometheus-fastapi-instrumentator==${PROMETHEUS_FASTAPI_INSTRUMENTATOR_VERSION}"
  export PATH="$HOME/.local/bin:$PATH"
  PYTHON_BIN="${PYTHON_BIN}"
  "${PYTHON_BIN}" bench/scripts/patch_vllm_metrics_compat.py
fi

if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
  pkill -f "vllm.entrypoints.openai.api_server" || true
  sleep 2
fi

LOG_FILE="/tmp/${RUN_ID}.vllm.log"
unset VLLM_VERSION
VLLM_ARGS=(
  --model "${MODEL}"
  --host "${VLLM_HOST}"
  --port "${PORT}"
  --dtype float16
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
)
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi
"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  "${VLLM_ARGS[@]}" \
  >"${LOG_FILE}" 2>&1 &
VLLM_PID=$!

cleanup() {
  kill "${VLLM_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

READY_URL="http://${VLLM_HOST}:${PORT}/v1/models"
echo "Waiting for vLLM readiness on ${READY_URL} ..."
for _ in $(seq 1 "${READINESS_TIMEOUT_SECS}"); do
  if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "${READY_URL}" >/dev/null 2>&1; then
  echo "vLLM did not become healthy. Log: ${LOG_FILE}"
  tail -n 120 "${LOG_FILE}" || true
  exit 1
fi

"${PYTHON_BIN}" bench/harness/run.py \
  --system vllm \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --vllm-version "${VLLM_VERSION_PIN}" \
  --scenarios "${SCENARIOS}" \
  --run-id "${RUN_ID}" \
  --gpu-type "${GPU_TYPE}" \
  --gpu-hour-usd "${GPU_HOUR_USD}" \
  --inference-mode real_model_inference \
  --determinism-check warn \
  --enable-gpu-sampling \
  --warmup-requests 1

echo "Baseline complete:"
echo "  bench/results/${RUN_ID}.csv"
echo "  bench/results/${RUN_ID}.manifest.json"
