# LLM Serving Runtime

Distributed transformer inference runtime with OpenAI-compatible API and reproducible benchmarking against vLLM.

## Design
- [Design notes](docs/design.md) summarize the systems choices behind Orca-style iteration-level scheduling, paged KV allocation, streaming retry boundaries, and coordinator routing.
- Real-inference smoke artifacts live in `bench/results/` for Phase 1 Transformers and vLLM on TinyLlama-1.1B.
- Generate the comparison chart with `python bench/scripts/plot_comparison.py`; output is written to `bench/results/comparison.png`.

![Real-inference smoke comparison](bench/results/comparison.png)

## Phase 0 status
- Repo scaffolded
- Benchmark harness skeleton + CSV/manifest schema
- Baseline scenarios including `mixed_concurrency` and `chat_multiturn`
- Determinism check (`temperature=0`) implemented in harness logic
- ADR pack drafted

## Quickstart
```bash
# Python harness env
python -m venv .venv
. .venv/Scripts/activate
pip install -r bench/harness/requirements.txt

# Validate scenario + schema
python bench/harness/run.py --dry-run --scenarios bench/scenarios/baseline.yaml
```

## Final benchmark handoff
- For pinned vLLM baseline generation on a supported Linux GPU host, run:
`bash scripts/gpu_host_handoff.sh`
- See full instructions:
`docs/reports/gpu-baseline-handoff.md`
