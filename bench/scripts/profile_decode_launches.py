"""Count cudaLaunchKernel events per decode step vs concurrency (read-only diagnostic)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PHASE2_BACKEND", "transformers")
os.environ.setdefault("HF_MODEL_ID", "Qwen/Qwen2-1.5B-Instruct")
os.environ.setdefault("HF_DEVICE", "cuda")


def _count_launches(kv_backend: str, num_active: int, decode_steps: int = 3) -> dict:
    os.environ["PHASE2_KV_BACKEND"] = kv_backend
    # Force re-init if backend already loaded
    import runtime.phase2.worker_server as ws

    ws._backend = None

    from runtime.phase2.kv_allocator import PagedKVAllocator
    from runtime.phase2.worker_server import ActiveState, GenerateRequest, TransformersBackend

    import torch
    from torch.profiler import ProfilerActivity, profile

    backend = TransformersBackend()
    alloc = PagedKVAllocator(
        total_blocks=512,
        block_size_tokens=16,
        bytes_per_token=backend.bytes_per_token,
    )

    states: list[ActiveState] = []
    for i in range(num_active):
        req = GenerateRequest(request_id=f"r{i}", prompt=f"hello world {i}", max_tokens=64)
        state = ActiveState(
            req=req,
            fut=None,
            stream_queue=None,
            words=[],
            generated=[],
            cursor=0,
            token_capacity=128,
            should_insert_prefix=False,
        )
        backend.init_state(state)
        allocation = alloc.allocate_for_tokens(req.request_id, 128)
        state.past_key_values = backend._create_kv_cache(128, allocation.block_ids)
        if backend.kv_backend == "reserved":
            state.contiguous_kv_hold = backend._allocate_contiguous_hold(128)
        backend._prefill_target(state)
        assert state.next_logits is not None
        state.last_token_id = int(state.next_logits.argmax(dim=-1).item())
        states.append(state)

    torch.cuda.synchronize()
    launches_per_step: list[int] = []

    for _ in range(decode_steps):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
            for state in states:
                backend.next_token(state)
        torch.cuda.synchronize()
        count = sum(
            1
            for evt in prof.key_averages()
            if "cudaLaunchKernel" in evt.key or "cudaLaunchKernel" in str(evt.key)
        )
        # Also count from events directly for reliability
        direct = sum(1 for e in prof.events() if e.name == "cudaLaunchKernel")
        launches_per_step.append(direct if direct else count)

    tokens_per_step = num_active
    avg_launches = sum(launches_per_step) / len(launches_per_step)
    return {
        "kv_backend": kv_backend,
        "num_active": num_active,
        "decode_steps_profiled": decode_steps,
        "tokens_per_step": tokens_per_step,
        "cudaLaunchKernel_per_step": avg_launches,
        "cudaLaunchKernel_per_token": avg_launches / max(1, tokens_per_step),
        "samples": launches_per_step,
    }


def main() -> None:
    import json

    from runtime.phase2.kv_allocator import PagedKVAllocator

    results = []
    for kv in ("reserved", "paged"):
        for n in (1, 8):
            print(f"profiling {kv} num_active={n} (sequential next_token)...", flush=True)
            results.append({**_count_launches(kv, n), "mode": "sequential_next_token"})

    # Batched path available for reserved only (StaticCache concat); not used by worker loop today.
    print("profiling reserved num_active=8 (next_token_batch)...", flush=True)
    os.environ["PHASE2_KV_BACKEND"] = "reserved"
    import runtime.phase2.worker_server as ws

    ws._backend = None
    from runtime.phase2.worker_server import ActiveState, GenerateRequest, TransformersBackend

    import torch
    from torch.profiler import ProfilerActivity, profile

    backend = TransformersBackend()
    alloc = PagedKVAllocator(
        total_blocks=512,
        block_size_tokens=16,
        bytes_per_token=backend.bytes_per_token,
    )
    states = []
    for i in range(8):
        req = GenerateRequest(request_id=f"b{i}", prompt=f"hello batch {i}", max_tokens=64)
        st = ActiveState(
            req=req,
            fut=None,
            stream_queue=None,
            words=[],
            generated=[],
            cursor=0,
            token_capacity=128,
            should_insert_prefix=False,
        )
        backend.init_state(st)
        a = alloc.allocate_for_tokens(req.request_id, 128)
        st.past_key_values = backend._create_kv_cache(128, a.block_ids)
        st.contiguous_kv_hold = backend._allocate_contiguous_hold(128)
        backend._prefill_target(st)
        assert st.next_logits is not None
        st.last_token_id = int(st.next_logits.argmax(dim=-1).item())
        states.append(st)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        backend.next_token_batch(states)
    torch.cuda.synchronize()
    batch_launches = sum(1 for e in prof.events() if e.name == "cudaLaunchKernel")
    results.append(
        {
            "kv_backend": "reserved",
            "num_active": 8,
            "mode": "next_token_batch",
            "tokens_per_step": 8,
            "cudaLaunchKernel_per_step": float(batch_launches),
            "cudaLaunchKernel_per_token": batch_launches / 8.0,
        }
    )
    print(json.dumps(results[-1], indent=2), flush=True)

    for row in results:
        print(json.dumps(row, indent=2), flush=True)

    out = ROOT / "bench" / "results" / "paged_kv_launch_profile.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
