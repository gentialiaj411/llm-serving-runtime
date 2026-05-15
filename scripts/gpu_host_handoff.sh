#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/gpu_host_handoff.sh
# Optional:
#   REPO_DIR=/workspace/orcaforge CONDA_ENV=orcaforge-vllm085 bash scripts/gpu_host_handoff.sh

REPO_DIR="${REPO_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-orcaforge-vllm085}"
PY_VER="${PY_VER:-3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.8.5}"
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
RUN_ID="${RUN_ID:-vllm-baseline-final}"
GPU_TYPE="${GPU_TYPE:-A100}"
GPU_HOUR_USD="${GPU_HOUR_USD:-3.40}"

cd "$REPO_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required on GPU host."
  exit 1
fi

eval "$(conda shell.bash hook)"
conda create -y -n "${CONDA_ENV}" "python=${PY_VER}" || true
conda activate "${CONDA_ENV}"

python -m pip install -U pip
python -m pip install -r bench/harness/requirements.txt
python -m pip install "vllm==${VLLM_VERSION}" "transformers==4.51.3" "tokenizers==0.21.1"

RUN_ID="${RUN_ID}" \
MODEL="${MODEL}" \
VLLM_VERSION="${VLLM_VERSION}" \
GPU_TYPE="${GPU_TYPE}" \
GPU_HOUR_USD="${GPU_HOUR_USD}" \
MAX_MODEL_LEN=1024 \
GPU_MEM_UTIL=0.70 \
bash scripts/run_vllm_baseline_ubuntu.sh

echo "Done. Artifacts:"
echo "  bench/results/${RUN_ID}.csv"
echo "  bench/results/${RUN_ID}.manifest.json"
