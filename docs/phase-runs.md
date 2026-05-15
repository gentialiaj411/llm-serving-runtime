# Phase Runs

## Phase 1
1. `python -m uvicorn frontend.phase1_server:app --host 127.0.0.1 --port 8000`
2. `python bench/harness/run.py --system phase1 --base-url http://127.0.0.1:8000/v1/chat/completions --run-id phase1-baseline`

## Phase 2
1. `python -m uvicorn runtime.phase2.worker_server:app --host 127.0.0.1 --port 8102`
2. `python -m uvicorn runtime.phase2.coordinator_server:app --host 127.0.0.1 --port 8000`
3. `python bench/harness/run.py --system phase2 --base-url http://127.0.0.1:8000/v1/chat/completions --run-id phase2-baseline`
