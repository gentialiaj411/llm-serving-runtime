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
GPU_TYPE="${GPU_TYPE:-unknown}"
GPU_HOUR_USD="${GPU_HOUR_USD:-2.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
SCENARIOS="${SCENARIOS:-bench/scenarios/baseline.yaml}"
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
  python -m pip install -q "vllm==${VLLM_VERSION}" "transformers==4.51.3" "tokenizers==0.21.1"
else
  "${PYTHON_BIN}" -m pip install -q --user --break-system-packages --upgrade pip
  "${PYTHON_BIN}" -m pip install -q --user --break-system-packages -r bench/harness/requirements.txt
  "${PYTHON_BIN}" -m pip install -q --user --break-system-packages "vllm==${VLLM_VERSION}" "transformers==4.51.3" "tokenizers==0.21.1"
  export PATH="$HOME/.local/bin:$PATH"
  PYTHON_BIN="${PYTHON_BIN}"
fi

if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
  pkill -f "vllm.entrypoints.openai.api_server" || true
  sleep 2
fi

LOG_FILE="/tmp/${RUN_ID}.vllm.log"
"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
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

echo "Waiting for vLLM health on ${VLLM_HOST}:${PORT} ..."
for _ in $(seq 1 240); do
  if curl -fsS "http://${VLLM_HOST}:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://${VLLM_HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "vLLM did not become healthy. Log: ${LOG_FILE}"
  tail -n 120 "${LOG_FILE}" || true
  exit 1
fi

"${PYTHON_BIN}" bench/harness/run.py \
  --system vllm \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --vllm-version "${VLLM_VERSION}" \
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
