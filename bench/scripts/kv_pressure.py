#!/usr/bin/env python3
"""KV memory pressure benchmark.

Simulates N concurrent inference requests at a random point in their lifecycle
and measures peak KV-cache memory under two allocation strategies:

  Paged KV   — only blocks for tokens *actually generated* are allocated
               (PagedKVAllocator with BLOCK_SIZE_TOKENS-token blocks)

  Contiguous — each request pre-reserves capacity for its full max_tokens
               budget upfront, as a naive contiguous allocator would

The difference shows the memory savings from demand-driven paged allocation.

Output JSON schema (written to --output):
  {
    "seed":                          <int>,
    "num_requests":                  <int>,
    "block_size_tokens":             <int>,
    "kv_bytes_per_token":            <int>,
    "peak_paged_bytes":              <int>,
    "peak_contiguous_baseline_bytes":<int>,
    "reduction_percent":             <float>
  }
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Support running directly from the repo root or from bench/scripts/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime.phase2.kv_allocator import PagedKVAllocator

# Heuristic KV-cache bytes per token for a typical 7B model at fp16:
# 2 (K+V) × 32 layers × 32 heads × 128 head-dim × 2 bytes
_KV_BYTES_PER_TOKEN = 2 * 32 * 32 * 128 * 2  # 524 288 bytes

_BLOCK_SIZE_TOKENS = 16


def _simulate(num_requests: int, seed: int) -> dict:
    rng = random.Random(seed)

    # Generate N requests each at a random point in generation.
    requests = []
    for i in range(num_requests):
        prompt_tokens = rng.randint(32, 512)
        max_gen_tokens = rng.randint(64, 256)
        # How far through generation is this request (5–95 % done).
        fraction_done = rng.uniform(0.05, 0.95)
        actual_gen_tokens = max(1, int(max_gen_tokens * fraction_done))
        requests.append(
            {
                "id": f"req-{i}",
                "current_tokens": prompt_tokens + actual_gen_tokens,
                "reserved_tokens": prompt_tokens + max_gen_tokens,
            }
        )

    # Paged KV: allocate blocks only for tokens generated so far.
    alloc = PagedKVAllocator(
        total_blocks=10_000_000,
        block_size_tokens=_BLOCK_SIZE_TOKENS,
        bytes_per_token=_KV_BYTES_PER_TOKEN,
    )
    peak_blocks = 0
    for req in requests:
        alloc.allocate_for_tokens(req["id"], req["current_tokens"])
        used = alloc.stats()["used_blocks"]
        if used > peak_blocks:
            peak_blocks = used

    peak_paged_bytes = peak_blocks * _BLOCK_SIZE_TOKENS * _KV_BYTES_PER_TOKEN

    # Contiguous baseline: every request reserves max_tokens upfront.
    peak_contiguous_baseline_bytes = (
        sum(req["reserved_tokens"] for req in requests) * _KV_BYTES_PER_TOKEN
    )

    reduction_percent = 0.0
    if peak_contiguous_baseline_bytes > 0:
        reduction_percent = (
            1.0 - peak_paged_bytes / peak_contiguous_baseline_bytes
        ) * 100.0

    return {
        "seed": seed,
        "num_requests": num_requests,
        "block_size_tokens": _BLOCK_SIZE_TOKENS,
        "kv_bytes_per_token": _KV_BYTES_PER_TOKEN,
        "peak_paged_bytes": peak_paged_bytes,
        "peak_contiguous_baseline_bytes": peak_contiguous_baseline_bytes,
        "reduction_percent": round(reduction_percent, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=64, help="Concurrent requests to simulate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = _simulate(args.requests, args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    paged_gb = result["peak_paged_bytes"] / 1e9
    cont_gb = result["peak_contiguous_baseline_bytes"] / 1e9
    print(
        f"requests={args.requests}  seed={args.seed}  "
        f"paged={paged_gb:.2f} GB  contiguous={cont_gb:.2f} GB  "
        f"reduction={result['reduction_percent']:.1f}%"
    )


if __name__ == "__main__":
    main()
