#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/bhask/Documents/PROJECTS/orcaforge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate awq-env
python -V
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
python -c 'import vllm; print(vllm.__version__)'
export RUN_ID=vllm-baseline-rtx5070-phase4
export MODEL=Qwen/Qwen2-1.5B-Instruct
export SCENARIOS=bench/results/phase2-rtx5070-throughput.scenario.yaml
export VLLM_VERSION=0.21.0
export GPU_TYPE='RTX 5070 Laptop GPU'
export GPU_HOUR_USD=0.80
export MAX_MODEL_LEN=1024
export GPU_MEM_UTIL=0.60
export VLLM_HOST=127.0.0.1
export PORT=8000
export VENV_DIR=$HOME/.venv-vllm-bench
bash scripts/run_vllm_baseline_ubuntu.sh