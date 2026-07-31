from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runtime.phase2.kv_allocator import PagedKVAllocator


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    timeout_step: int

    @property
    def token_capacity(self) -> int:
        return self.prompt_tokens + self.output_tokens


def make_workload(count: int, seed: int) -> list[Request]:
    rng = random.Random(seed)
    return [
        Request(
            request_id=f"req-{i}",
            prompt_tokens=rng.randint(32, 1024),
            output_tokens=rng.randint(16, 256),
            timeout_step=rng.randint(20, 80),
        )
        for i in range(count)
    ]


def run_pressure(
    requests: int,
    seed: int,
    bytes_per_token: int,
    block_size_tokens: int,
    max_active: int,
) -> dict[str, int | float]:
    workload = make_workload(requests, seed)
    total_tokens_ever_needed = sum(req.token_capacity for req in workload)
    total_blocks = max(1, total_tokens_ever_needed // block_size_tokens + requests)
    allocator = PagedKVAllocator(
        total_blocks=total_blocks,
        block_size_tokens=block_size_tokens,
        bytes_per_token=bytes_per_token,
    )

    waiting = list(workload)
    active: dict[str, tuple[Request, int]] = {}
    peak_paged_bytes = 0
    step = 0

    while waiting or active:
        while waiting and len(active) < max_active:
            req = waiting.pop(0)
            alloc = allocator.allocate_for_tokens(req.request_id, req.token_capacity)
            if alloc is None:
                waiting.insert(0, req)
                break
            active[req.request_id] = (req, req.output_tokens)

        stats = allocator.stats()
        peak_paged_bytes = max(peak_paged_bytes, int(stats["used_blocks"]) * int(stats["block_size_bytes"]))

        finished: list[str] = []
        for request_id, (req, remaining) in active.items():
            remaining -= 16
            if remaining <= 0 or step >= req.timeout_step:
                finished.append(request_id)
            else:
                active[request_id] = (req, remaining)

        for request_id in finished:
            active.pop(request_id, None)
            allocator.free_request(request_id)
        step += 1

    contiguous_baseline_bytes = total_tokens_ever_needed * bytes_per_token
    reduction = 100.0 * (1.0 - (peak_paged_bytes / contiguous_baseline_bytes))
    return {
        "peak_paged_bytes": peak_paged_bytes,
        "peak_contiguous_baseline_bytes": contiguous_baseline_bytes,
        "reduction_percent": reduction,
        "baseline_definition": "contiguous_baseline_bytes = sum(prompt_tokens + output_tokens) * bytes_per_token",
        "workload_model": "synthetic_uniform_prompt_output_with_timeout",
        "request_token_capacity_definition": "prompt_tokens + output_tokens",
        "deterministic": True,
        "seed": seed,
        "request_count": requests,
        "bytes_per_token": bytes_per_token,
        "block_size_tokens": block_size_tokens,
        "max_active": max_active,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5070)
    parser.add_argument("--bytes-per-token", type=int, default=262144)
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--max-active", type=int, default=16)
    parser.add_argument("--output", default="bench/results/kv-pressure.json")
    args = parser.parse_args()

    result = run_pressure(
        requests=args.requests,
        seed=args.seed,
        bytes_per_token=args.bytes_per_token,
        block_size_tokens=args.block_size_tokens,
        max_active=args.max_active,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
