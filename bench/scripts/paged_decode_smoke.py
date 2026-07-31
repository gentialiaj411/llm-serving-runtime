"""Quick smoke timing for paged decode path."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PHASE2_BACKEND", "transformers")
os.environ.setdefault("PHASE2_KV_BACKEND", "paged")
os.environ.setdefault("HF_MODEL_ID", "Qwen/Qwen2-1.5B-Instruct")
os.environ.setdefault("HF_DEVICE", "cuda")

from runtime.phase2.kv_allocator import PagedKVAllocator
from runtime.phase2.paged_attention_triton import triton_available
from runtime.phase2.worker_server import ActiveState, GenerateRequest, TransformersBackend


def main() -> None:
    print("triton_available:", triton_available())
    t0 = time.perf_counter()
    backend = TransformersBackend()
    print(f"loaded backend in {time.perf_counter() - t0:.1f}s")
    req = GenerateRequest(request_id="x", prompt="hello world", max_tokens=32)
    state = ActiveState(
        req=req,
        fut=None,
        stream_queue=None,
        words=[],
        generated=[],
        cursor=0,
        token_capacity=64,
        should_insert_prefix=False,
    )
    backend.init_state(state)
    alloc = PagedKVAllocator(
        total_blocks=512,
        block_size_tokens=16,
        bytes_per_token=backend.bytes_per_token,
    )
    allocation = alloc.allocate_for_tokens("x", 64)
    state.past_key_values = backend._create_kv_cache(64, allocation.block_ids)
    backend._prefill_target(state)
    print(f"prefill done in {time.perf_counter() - t0:.1f}s")
    start = time.perf_counter()
    for _ in range(16):
        backend.next_token(state)
    elapsed = time.perf_counter() - start
    print(f"16 decode tokens in {elapsed:.2f}s ({16 / elapsed:.2f} tok/s)")


if __name__ == "__main__":
    main()
