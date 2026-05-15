#!/usr/bin/env bash
set -euo pipefail
python bench/harness/run.py --dry-run --scenarios bench/scenarios/baseline.yaml
