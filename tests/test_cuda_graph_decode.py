"""Parity + microbench: eager vs CUDA-graph paged decode (PHASE2_CUDA_GRAPH)."""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class CudaGraphSmokeTests(unittest.TestCase):
    @unittest.skipUnless(__import__("torch").cuda.is_available(), "CUDA required")
    def test_cuda_graph_linear_smoke(self) -> None:
        from runtime.phase2.cuda_graph_decode import capture_replay_linear_smoke

        result = capture_replay_linear_smoke()
        self.assertTrue(result.get("ok"), msg=str(result))


def _build_backend(cuda_graph: bool):
    os.environ["PHASE2_BACKEND"] = "transformers"
    os.environ["PHASE2_KV_BACKEND"] = "paged"
    os.environ["HF_MODEL_ID"] = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2-1.5B-Instruct")
    os.environ["HF_DEVICE"] = "cuda"
    os.environ["PHASE2_CUDA_GRAPH"] = "1" if cuda_graph else "0"
    os.environ["PHASE2_CUDA_GRAPH_WARMUP"] = "2"
    os.environ["PHASE2_MAX_ACTIVE"] = "8"

    import runtime.phase2.worker_server as ws

    ws._backend = None
    from runtime.phase2.kv_allocator import PagedKVAllocator
    from runtime.phase2.worker_server import ActiveState, GenerateRequest, TransformersBackend

    backend = TransformersBackend()
    alloc = PagedKVAllocator(1024, 16, bytes_per_token=backend.bytes_per_token)
    return backend, alloc, ActiveState, GenerateRequest


def _make_states(backend, alloc, ActiveState, GenerateRequest, n: int, seed: int = 123):
    import torch

    torch.manual_seed(seed)
    states = []
    for i in range(n):
        req = GenerateRequest(
            request_id=f"g{i}",
            prompt=f"cuda graph parity prompt {i} " * 6,
            max_tokens=32,
        )
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
        state.generated_token_ids = []
        states.append(state)
    return states


def _decode_tokens(backend, states, steps: int) -> list[list[int]]:
    import torch

    for _ in range(steps):
        backend._paged_decode_step(states, getattr(backend, "_paged_persistent", None))
    torch.cuda.synchronize()
    return [list(s.generated_token_ids or []) for s in states]


class CudaGraphDecodeParityTests(unittest.TestCase):
    @unittest.skipUnless(__import__("torch").cuda.is_available(), "CUDA required")
    def test_graphed_flag_matches_eager_token_ids(self) -> None:
        """Eager vs PHASE2_CUDA_GRAPH=1 must agree on token ids (capture or eager fallback)."""
        steps = 8
        batch = 2

        eager_backend, eager_alloc, ActiveState, GenerateRequest = _build_backend(False)
        eager_states = _make_states(eager_backend, eager_alloc, ActiveState, GenerateRequest, batch)
        eager_ids = _decode_tokens(eager_backend, eager_states, steps)

        graph_backend, graph_alloc, ActiveState, GenerateRequest = _build_backend(True)
        graph_states = _make_states(graph_backend, graph_alloc, ActiveState, GenerateRequest, batch)
        graph_ids = _decode_tokens(graph_backend, graph_states, steps)

        self.assertEqual(eager_ids, graph_ids)

        mgr = getattr(graph_backend, "_cuda_graph_manager", None)
        self.assertIsNotNone(mgr)
        stats = mgr.stats()
        # Record-only: HF capture may be blocked; smoke test covers CUDAGraph itself.
        print("cuda_graph_stats", json.dumps(stats), flush=True)


def _run_comparison() -> dict:
    import torch

    from bench.scripts._cuda_op_counter import CudaOpCounter
    from runtime.phase2.cuda_graph_decode import capture_replay_linear_smoke

    smoke = capture_replay_linear_smoke()
    steps = 12
    batch = 2
    rows = []

    for label, enabled in (("eager", False), ("cuda_graph", True)):
        backend, alloc, ActiveState, GenerateRequest = _build_backend(enabled)
        states = _make_states(backend, alloc, ActiveState, GenerateRequest, batch, seed=7)
        _decode_tokens(backend, states, 4)  # warmup / capture attempt

        torch.cuda.synchronize()
        counter = CudaOpCounter()
        t0 = time.perf_counter()
        with counter:
            _decode_tokens(backend, states, steps)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = batch * steps
        mgr = getattr(backend, "_cuda_graph_manager", None)
        rows.append(
            {
                "mode": label,
                "PHASE2_CUDA_GRAPH": enabled,
                "batch_size": batch,
                "decode_steps": steps,
                "tokens": tokens,
                "wall_s": elapsed,
                "tok_per_s": tokens / max(elapsed, 1e-9),
                "cuda_op_proxy_total": counter.total,
                "cuda_op_proxy_per_token": counter.total / max(tokens, 1),
                "graph_stats": mgr.stats() if mgr is not None else None,
            }
        )

    eager = rows[0]
    graphed = rows[1]
    captured = False
    if graphed.get("graph_stats"):
        captured = any(b.get("captured") for b in graphed["graph_stats"].get("buckets", []))

    return {
        "model_id": os.environ.get("HF_MODEL_ID", "Qwen/Qwen2-1.5B-Instruct"),
        "metric_note": "cuda_op_proxy is TorchDispatchMode CUDA aten-op count (CUPTI often unavailable)",
        "linear_cudagraph_smoke": smoke,
        "hf_paged_graph_captured": captured,
        "rows": rows,
        "delta": {
            "tok_per_s": graphed["tok_per_s"] - eager["tok_per_s"],
            "tok_per_s_ratio": graphed["tok_per_s"] / max(eager["tok_per_s"], 1e-9),
            "ops_per_token_ratio": graphed["cuda_op_proxy_per_token"]
            / max(eager["cuda_op_proxy_per_token"], 1e-9),
        },
        "verdict": {
            "linear_smoke_ok": bool(smoke.get("ok")),
            "hf_capture_ok": captured,
            "parity_path": "eager_fallback" if not captured else "graph_replay",
            "throughput_improved": graphed["tok_per_s"] > eager["tok_per_s"] and captured,
        },
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--bench":
        payload = _run_comparison()
        out = ROOT / "bench" / "results" / "cuda_graph_comparison.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        print(f"wrote {out}")
    else:
        unittest.main()
