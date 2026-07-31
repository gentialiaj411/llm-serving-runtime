#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for ma in 1 2 4 8; do
  echo "=== max_active=${ma} ==="
  python3 bench/scripts/paged_kv_real_gpu.py \
    --requests 32 \
    --max-active "${ma}" \
    --output-json "bench/results/paged_kv_scaling_ma${ma}.json" \
    --output-md "bench/results/paged_kv_scaling_ma${ma}.md"
done
