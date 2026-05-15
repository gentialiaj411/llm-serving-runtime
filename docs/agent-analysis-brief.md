# LLM Agent Analysis Brief: Orcaforge (Distributed Transformer Inference Runtime)

## 1) Context and Goal
You are analyzing a multi-month ML systems project: a distributed transformer inference serving runtime intended to be resume-grade for ML infra recruiting.

Primary objective:
- Evaluate the current codebase quality and implementation status.
- Identify architectural and implementation risks.
- Produce a concrete, prioritized plan to reach a credible end-state benchmarked against vLLM.

This is an inference-systems project, not model research.

## 2) Project Summary
Orcaforge is intended to serve Llama-family models with:
- OpenAI-compatible HTTP API
- Coordinator + GPU worker architecture
- Continuous batching and scheduling
- Paged KV cache management
- Fault-tolerance primitives (health checks, cancellation, replay-safe recovery)
- Benchmark harness for throughput/latency comparisons vs vLLM

Current development references:
- Fast iteration model: TinyLlama-1.1B
- Headline benchmark model: Llama-3-8B
- Baseline: vLLM on same hardware

## 3) Target Architecture (Intended)
- Frontend: Python/FastAPI OpenAI-compatible endpoints (`/v1/chat/completions`, etc.)
- Coordinator: C++ process for admission control, queueing, routing, deadlines, cancellation
- Workers: C++/CUDA process per GPU for batching, scheduling, KV cache, kernels
- Inter-process transport: gRPC (minimal hot-path overhead)
- Deployment: Docker (+ K8s manifests for multi-worker demo)

## 4) Reality Check (What Exists Now)
The repo currently contains substantial Python runtime stand-ins and harness tooling, plus C++ skeleton/toolchain setup. Treat this as an in-progress systems codebase with mixed maturity.

Known status:
- Harness exists and emits CSV + manifest.
- Determinism and scenario coverage were recently improved.
- Chaos test tooling exists.
- vLLM pinned baseline artifact is still incomplete locally due to GPU/toolchain compatibility constraints.

## 5) What To Analyze
Please inspect the entire repo and produce a critical technical review covering:

1. Architecture fidelity
- How close implementation is to intended C++/CUDA + gRPC architecture.
- What is still prototype-level vs production-level.

2. Performance methodology
- Benchmark harness correctness (scenario execution, metrics semantics, determinism checks).
- Fairness of vLLM comparison methodology.
- Any measurement blind spots or invalid assumptions.

3. Reliability and fault tolerance
- Correctness of deadline handling, cancellation, retries/replay, and health-based routing.
- Failure modes likely to break SLA under load/chaos.

4. KV cache and scheduler design
- Whether allocator/scheduler interfaces are robust enough for later CUDA integration.
- Fragmentation metrics usefulness and missing instrumentation.

5. Code quality and maintainability
- Module boundaries, testability, observability, config hygiene, and ADR traceability.
- Technical debt likely to slow Phase 3+ work.

## 6) Required Output Format
Return your findings in this exact structure:

### A. Executive Summary (10-15 lines)
- Current maturity level
- Top 3 blockers to credible benchmark claims
- Whether this is currently “interview defensible” for ML infra roles

### B. Findings (Severity-ordered)
For each finding include:
- Severity: Critical / High / Medium / Low
- File references (path + line numbers where possible)
- Why it matters
- Suggested fix

### C. Gap-to-Goal Matrix
A table with columns:
- Goal capability
- Current status
- Evidence in repo
- Gap
- Effort (S/M/L)
- Risk

Include at least:
- OpenAI API compatibility
- Continuous batching correctness
- Paged KV allocator realism
- Fault-tolerant routing and recovery
- Reproducible vLLM baseline
- C++ coordinator readiness
- C++/CUDA worker readiness

### D. 30/60/90 Day Technical Plan
- Concrete milestones with acceptance artifacts.
- Explicit benchmark/chaos artifacts required at each milestone.
- Include dependency ordering (what must happen before what).

### E. Resume Claim Readiness
For each intended resume bullet:
- Claim text (draft)
- Evidence currently present
- Evidence missing
- Confidence (0-100%)

## 7) Constraints and Review Style
- Be direct and critical; avoid generic encouragement.
- Prefer concrete file-level evidence over abstract advice.
- If uncertain, state uncertainty explicitly.
- Do not assume missing benchmark artifacts exist.
- Call out anything that looks “resume-inflated” vs currently evidenced.

## 8) Important Repo Paths To Start From
- `README.md`
- `docs/phase-status.md`
- `docs/runtime-status.md`
- `docs/resume-bullets.md`
- `docs/adr/`
- `runtime/`
- `bench/harness/run.py`
- `bench/scenarios/baseline.yaml`
- `bench/chaos/run_chaos.py`
- `bench/results/`
- `.github/workflows/ci.yml`

## 9) Primary Question To Answer
If this codebase were submitted as a portfolio project for ML inference/systems internships, what would a strong infra engineer trust, and what would they challenge immediately?

