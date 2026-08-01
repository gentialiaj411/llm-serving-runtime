"""Attribute GPU-dispatching aten ops inside one paged decode step by call site.

Uses TorchDispatchMode as a CUPTI-free proxy for kernel launches (RTX 50xx /
Windows often lacks working CUPTI in torch.profiler).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PHASE2_BACKEND", "transformers")
os.environ.setdefault("HF_MODEL_ID", "Qwen/Qwen2-1.5B-Instruct")
os.environ.setdefault("HF_DEVICE", "cuda")
os.environ["PHASE2_KV_BACKEND"] = "paged"

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
        backend._prefill_target(state)
        assert state.next_logits is not None
        state.last_token_id = int(state.next_logits.argmax(dim=-1).item())
        state.prompt_prefilled = True
        states.append(state)
    return states


def _count(fn) -> tuple[int, dict[str, int]]:
    import torch

    torch.cuda.synchronize()
    counter = CudaOpCounter()
    with counter:
        fn()
    torch.cuda.synchronize()
    return counter.total, dict(counter.by_op.most_common(15))


def main() -> None:
    import torch

    import runtime.phase2.worker_server as ws
    from runtime.phase2.kv_allocator import PagedKVAllocator
    from runtime.phase2.worker_server import TransformersBackend

    ws._backend = None
    backend = TransformersBackend()
    alloc = PagedKVAllocator(
        total_blocks=1024,
        block_size_tokens=16,
        bytes_per_token=backend.bytes_per_token,
    )

    attribution: dict = {
        "model_id": os.environ["HF_MODEL_ID"],
        "kv_backend": "paged",
        "method": "TorchDispatchMode CUDA aten-op counts (CUPTI unavailable on this host)",
        "cupti_status": "unavailable",
        "rows": [],
    }

    for num_active in (1, 8):
        print(f"attributing num_active={num_active}", flush=True)
        states = _make_states(backend, alloc, num_active)
        # Warmup
        if num_active == 1:
            backend._paged_decode_step(states, None)
        else:
            batch = backend._concat_paged_caches(states)
            _, batch = backend._paged_decode_step(states, batch)
            backend._split_paged_batch_cache(batch, states)
        torch.cuda.synchronize()

        sites: dict[str, object] = {}
        if num_active == 1:
            n, top = _count(lambda: backend._paged_decode_step(states, None))
            sites["site:model_forward_single"] = {"launches_proxy": n, "top_ops": top}
            sites["site:concat_paged_caches"] = {"launches_proxy": 0, "top_ops": {}}
            primary = n
        else:
            n, top = _count(lambda: backend._concat_paged_caches(states))
            sites["site:concat_paged_caches"] = {"launches_proxy": n, "top_ops": top}
            batch = backend._concat_paged_caches(states)

            n, top = _count(lambda: backend._paged_decode_step(states, batch))
            sites["site:model_forward_batched"] = {"launches_proxy": n, "top_ops": top}
            primary = n

            n, top = _count(lambda: backend._paged_decode_step(states, None))
            sites["site:decode_with_rebuild"] = {"launches_proxy": n, "top_ops": top}

            n, top = _count(lambda: backend._split_paged_batch_cache(batch, states))
            sites["site:split_paged_batch_cache"] = {"launches_proxy": n, "top_ops": top}

            # Recreate states' caches after split for sequential path fairness
            n, top = _count(lambda: [backend.next_token(s) for s in states])
            sites["site:sequential_next_token"] = {"launches_proxy": n, "top_ops": top}

        row = {
            "num_active": num_active,
            "call_site_launch_counts": sites,
            "primary_decode_launches_proxy": primary,
        }
        attribution["rows"].append(row)
        print(json.dumps({"num_active": num_active, "primary": primary, "sites": {k: v["launches_proxy"] for k, v in sites.items()}}, indent=2), flush=True)

        for state in states:
            alloc.free_request(state.req.request_id)

    b1 = int(attribution["rows"][0]["primary_decode_launches_proxy"])
    sites8 = attribution["rows"][1]["call_site_launch_counts"]
    b8 = int(sites8["site:model_forward_batched"]["launches_proxy"])
    concat = int(sites8["site:concat_paged_caches"]["launches_proxy"])
    seq = int(sites8["site:sequential_next_token"]["launches_proxy"])
    attribution["verdict"] = {
        "concat_is_dominant_launch_source": concat > 0.2 * b8 if b8 else False,
        "concat_launches_proxy_at_b8": concat,
        "batched_forward_launches_proxy_b1": b1,
        "batched_forward_launches_proxy_b8": b8,
        "batched_launch_growth_b1_to_b8": (b8 / b1) if b1 else None,
        "sequential_launches_proxy_b8": seq,
        "sequential_explains_old_8x_curve": bool(b1 and seq and abs(seq / b1 - 8.0) < 1.5),
        "target_growth_le_2x": (b8 / b1) <= 2.0 if b1 else False,
        "prime_suspect_confirmed": False,
        "prime_suspect_note": (
            "_concat_paged_caches only builds block_tables/seq_lens tensors; it does not "
            "torch.cat KV rows. Old 8x curve came from sequential next_token profiling."
        ),
    }

    out = ROOT / "bench" / "results" / "launch_attribution.json"
    out.write_text(json.dumps(attribution, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print(json.dumps(attribution["verdict"], indent=2), flush=True)


if __name__ == "__main__":
    main()
