"""Count decode dispatch ops vs concurrency (batched path; CUPTI-free proxy)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PHASE2_BACKEND", "transformers")
os.environ.setdefault("HF_MODEL_ID", "Qwen/Qwen2-1.5B-Instruct")
os.environ.setdefault("HF_DEVICE", "cuda")

from bench.scripts._cuda_op_counter import CudaOpCounter  # noqa: E402


def _make_states(backend, alloc, n: int):
    from runtime.phase2.worker_server import ActiveState, GenerateRequest

    states = []
    for i in range(n):
        req = GenerateRequest(request_id=f"r{i}", prompt=f"hello world {i} " * 8, max_tokens=64)
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
        assert allocation is not None
        state.past_key_values = backend._create_kv_cache(128, allocation.block_ids)
        if backend.kv_backend == "reserved":
            state.contiguous_kv_hold = backend._allocate_contiguous_hold(128)
        backend._prefill_target(state)
        assert state.next_logits is not None
        state.last_token_id = int(state.next_logits.argmax(dim=-1).item())
        state.prompt_prefilled = True
        states.append(state)
    return states


def _count_batched(kv_backend: str, num_active: int, decode_steps: int = 3) -> dict:
    os.environ["PHASE2_KV_BACKEND"] = kv_backend
    import runtime.phase2.worker_server as ws

    ws._backend = None
    from runtime.phase2.kv_allocator import PagedKVAllocator
    from runtime.phase2.worker_server import TransformersBackend

    import torch

    backend = TransformersBackend()
    alloc = PagedKVAllocator(
        total_blocks=1024,
        block_size_tokens=16,
        bytes_per_token=backend.bytes_per_token,
    )
    states = _make_states(backend, alloc, num_active)

    def _one_step() -> None:
        if kv_backend == "paged":
            backend._paged_decode_step(states, getattr(backend, "_paged_persistent", None))
        elif num_active == 1:
            backend.next_token(states[0])
        else:
            # Reserved StaticCache does not support legacy concat; measure sequential.
            for state in states:
                backend.next_token(state)

    # Warmup
    _one_step()
    torch.cuda.synchronize()

    samples: list[int] = []
    wall_ms: list[float] = []
    for _ in range(decode_steps):
        torch.cuda.synchronize()
        counter = CudaOpCounter()
        t0 = time.perf_counter()
        with counter:
            _one_step()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000.0)
        samples.append(counter.total)

    avg = sum(samples) / len(samples)
    mode = (
        "batched_paged_decode_step"
        if kv_backend == "paged"
        else ("single_next_token" if num_active == 1 else "sequential_next_token_reserved")
    )
    return {
        "kv_backend": kv_backend,
        "num_active": num_active,
        "mode": mode,
        "decode_steps_profiled": decode_steps,
        "tokens_per_step": num_active,
        "cuda_op_proxy_per_step": avg,
        "cuda_op_proxy_per_token": avg / max(1, num_active),
        "cudaLaunchKernel_per_step": avg,
        "cudaLaunchKernel_per_token": avg / max(1, num_active),
        "wall_ms_per_step_mean": sum(wall_ms) / len(wall_ms),
        "samples": samples,
        "metric": "TorchDispatchMode_cuda_aten_ops",
    }


def main() -> None:
    from runtime.phase2.paged_attention_triton import triton_available

    results = []
    before_after = {}
    for kv in ("reserved", "paged"):
        row1 = None
        for n in (1, 8):
            print(f"profiling {kv} num_active={n} (batched)...", flush=True)
            row = _count_batched(kv, n)
            results.append(row)
            print(json.dumps(row, indent=2), flush=True)
            if n == 1:
                row1 = row
            elif row1 is not None:
                growth = row["cudaLaunchKernel_per_step"] / max(1e-9, row1["cudaLaunchKernel_per_step"])
                before_after[kv] = {
                    "b1_ops_per_step": row1["cudaLaunchKernel_per_step"],
                    "b8_ops_per_step": row["cudaLaunchKernel_per_step"],
                    "growth_b1_to_b8": growth,
                    "b1_wall_ms": row1["wall_ms_per_step_mean"],
                    "b8_wall_ms": row["wall_ms_per_step_mean"],
                    "b8_tok_per_s_proxy": (8.0 / (row["wall_ms_per_step_mean"] / 1000.0))
                    if row["wall_ms_per_step_mean"]
                    else None,
                }

    # Historical sequential baseline (explains old 8x curve).
    print("profiling paged num_active=8 sequential next_token (legacy methodology)...", flush=True)
    os.environ["PHASE2_KV_BACKEND"] = "paged"
    import runtime.phase2.worker_server as ws

    ws._backend = None
    from runtime.phase2.kv_allocator import PagedKVAllocator
    from runtime.phase2.worker_server import TransformersBackend
    import torch

    backend = TransformersBackend()
    alloc = PagedKVAllocator(1024, 16, bytes_per_token=backend.bytes_per_token)
    states = _make_states(backend, alloc, 8)
    torch.cuda.synchronize()
    counter = CudaOpCounter()
    with counter:
        for state in states:
            backend.next_token(state)
    torch.cuda.synchronize()
    results.append(
        {
            "kv_backend": "paged",
            "num_active": 8,
            "mode": "sequential_next_token_legacy",
            "cudaLaunchKernel_per_step": float(counter.total),
            "cudaLaunchKernel_per_token": counter.total / 8.0,
            "metric": "TorchDispatchMode_cuda_aten_ops",
        }
    )

    artifact = {
        "timestamp_note": "CUPTI unavailable on native Windows/sm_120; using CUDA aten-op proxy",
        "triton_available": triton_available(),
        "scaling": before_after,
        "acceptance": {
            "target_growth_le_2x": {
                kv: (info["growth_b1_to_b8"] <= 2.0) for kv, info in before_after.items()
            }
        },
        "rows": results,
    }

    # Flat list compatibility with prior consumers + rich header.
    out = ROOT / "bench" / "results" / "paged_kv_launch_profile.json"
    # Prior file was a bare list; keep list at top-level for compatibility, write companion meta.
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    meta_out = ROOT / "bench" / "results" / "paged_kv_launch_profile.meta.json"
    meta_out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print(f"wrote {meta_out}", flush=True)
    print(json.dumps(artifact["scaling"], indent=2), flush=True)
    print(json.dumps(artifact["acceptance"], indent=2), flush=True)


if __name__ == "__main__":
    main()
