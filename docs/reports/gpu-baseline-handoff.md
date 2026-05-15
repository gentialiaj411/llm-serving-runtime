# GPU Baseline Handoff (Final Missing Artifact)

This project is functionally complete locally. The remaining missing artifact is a pinned vLLM baseline run on a supported GPU host.

## Target artifacts
- `bench/results/vllm-baseline-final.csv`
- `bench/results/vllm-baseline-final.manifest.json`

## Why this is needed
- Resume comparisons must be measured against pinned `vllm==0.8.5`.
- Local RTX 5070 laptop path failed with CUDA kernel compatibility (`sm_120`) for this pinned stack.

## One-command run on GPU host
From repo root on an A100/H100-compatible Linux host:

```bash
bash scripts/gpu_host_handoff.sh
```

Optional overrides:

```bash
REPO_DIR=$PWD CONDA_ENV=orcaforge-vllm085 PY_VER=3.12 VLLM_VERSION=0.8.5 \
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 RUN_ID=vllm-baseline-final \
GPU_TYPE=A100 GPU_HOUR_USD=3.40 \
bash scripts/gpu_host_handoff.sh
```

## Copy artifacts back
Copy these files into this repo's `bench/results/` directory and commit:
- `vllm-baseline-final.csv`
- `vllm-baseline-final.manifest.json`

## Completion definition
Project is benchmark-complete when:
1. Runtime CSV + manifest exist.
2. Chaos SLA JSON exists.
3. vLLM baseline CSV + manifest exist (pinned version recorded in manifest).
