"""Prefill vs decode wall-time breakdown: dynamic KV vs paged KV.

Decode-heavy shape (default 128 prompt / 512 decode) matches ablation_decode_heavy.

Writes: bench/results/paged_vs_dynamic_breakdown.json

Reproduce:
  .venv311\\Scripts\\python.exe bench/scripts/paged_vs_dynamic_breakdown.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _pad_prompt(tokenizer: Any, target_tokens: int) -> str:
    chunk = "Orcaforge decode-heavy timing prompt. Measure prefill versus decode. "
    text = chunk
    while True:
        n = int(tokenizer(text, return_tensors="pt")["input_ids"].shape[1])
        if n >= target_tokens:
            break
        text += chunk
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0, :target_tokens].tolist()
    return tokenizer.decode(ids, skip_special_tokens=True)


def _run_backend(
    *,
    kv_backend: str,
    model_id: str,
    prompt_tokens: int,
    decode_tokens: int,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    os.environ["PHASE2_BACKEND"] = "transformers"
    os.environ["PHASE2_KV_BACKEND"] = kv_backend
    os.environ["HF_MODEL_ID"] = model_id
    os.environ.setdefault("HF_DEVICE", "cuda")
    os.environ.setdefault("HF_TORCH_DTYPE", "float16")
    os.environ.setdefault("KV_TOTAL_BLOCKS", "4096")
    os.environ.setdefault("KV_BLOCK_SIZE_TOKENS", "16")
    os.environ["PHASE2_PREFIX_CACHE"] = "0"
    os.environ["PHASE2_CUDA_GRAPH"] = "0"

    import runtime.phase2.worker_server as ws
    from runtime.phase2.kv_allocator import PagedKVAllocator
    from runtime.phase2.worker_server import ActiveState, GenerateRequest, TransformersBackend

    ws._backend = None
    backend = TransformersBackend()
    prompt = _pad_prompt(backend.tokenizer, prompt_tokens)

    def _make_states(n: int) -> tuple[list[Any], PagedKVAllocator]:
        alloc = PagedKVAllocator(
            total_blocks=4096,
            block_size_tokens=int(os.environ.get("KV_BLOCK_SIZE_TOKENS", "16")),
            bytes_per_token=backend.bytes_per_token,
        )
        states = []
        for i in range(n):
            req = GenerateRequest(
                request_id=f"{kv_backend}-{i}",
                prompt=prompt,
                max_tokens=decode_tokens,
            )
            state = ActiveState(
                req=req,
                fut=None,  # type: ignore[arg-type]
                stream_queue=None,
                words=[],
                generated=[],
                cursor=0,
                token_capacity=prompt_tokens + decode_tokens + 8,
                should_insert_prefix=False,
            )
            backend.init_state(state)
            allocation = alloc.allocate_for_tokens(req.request_id, state.token_capacity)
            assert allocation is not None
            if kv_backend == "paged":
                state.past_key_values = backend._create_kv_cache(
                    state.token_capacity, allocation.block_ids
                )
            states.append(state)
        return states, alloc

    def _time_prefill(states: list[Any]) -> float:
        _sync()
        t0 = time.perf_counter()
        if kv_backend == "paged":
            backend._paged_prefill_groups(states)
        else:
            # Use serving batched prefill (emits first token), matching paged path.
            backend.next_token_batch(states)
        _sync()
        return time.perf_counter() - t0

    def _time_decode(states: list[Any], steps: int) -> float:
        # Prefill path already emitted the first token for both backends.
        remaining = max(0, steps - 1)
        if remaining <= 0:
            return 0.0
        _sync()
        t0 = time.perf_counter()
        if kv_backend == "paged":
            batch = None
            for _ in range(remaining):
                _, batch = backend._paged_decode_step(states, batch)
        else:
            for _ in range(remaining):
                backend.next_token_batch(states)
        _sync()
        return time.perf_counter() - t0

    # Warmup (discard)
    for _ in range(warmup):
        states, _alloc = _make_states(batch_size)
        _time_prefill(states)
        _time_decode(states, min(8, decode_tokens))

    prefill_samples: list[float] = []
    decode_samples: list[float] = []
    for _ in range(repeats):
        states, _alloc = _make_states(batch_size)
        prefill_s = _time_prefill(states)
        decode_s = _time_decode(states, decode_tokens)
        prefill_samples.append(prefill_s)
        decode_samples.append(decode_s)

    def _median(xs: list[float]) -> float:
        ys = sorted(xs)
        return ys[len(ys) // 2]

    prefill_med = _median(prefill_samples)
    decode_med = _median(decode_samples)
    total = prefill_med + decode_med
    decode_tok = batch_size * decode_tokens
    return {
        "kv_backend": kv_backend,
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "warmup": warmup,
        "repeats": repeats,
        "prefill_wall_s_samples": prefill_samples,
        "decode_wall_s_samples": decode_samples,
        "prefill_wall_s_median": prefill_med,
        "decode_wall_s_median": decode_med,
        "total_wall_s_median": total,
        "prefill_fraction": prefill_med / max(1e-9, total),
        "decode_fraction": decode_med / max(1e-9, total),
        "decode_tok_per_s_median": decode_tok / max(1e-9, decode_med),
        "end_to_end_tok_per_s_median": decode_tok / max(1e-9, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-json", default="bench/results/paged_vs_dynamic_breakdown.json"
    )
    args = parser.parse_args()

    rows = []
    for backend in ("dynamic", "paged"):
        print(f"running breakdown kv_backend={backend}", flush=True)
        rows.append(
            _run_backend(
                kv_backend=backend,
                model_id=args.model_id,
                prompt_tokens=args.prompt_tokens,
                decode_tokens=args.decode_tokens,
                batch_size=args.batch_size,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )

    by_name = {r["kv_backend"]: r for r in rows}
    dyn = by_name["dynamic"]
    paged = by_name["paged"]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_type": "paged_vs_dynamic_breakdown",
        "model_id": args.model_id,
        "scenario": {
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "batch_size": args.batch_size,
            "note": "decode-heavy; wall times include CUDA synchronize",
        },
        "measurement": "measured_on_gpu",
        "backends": rows,
        "comparison": {
            "paged_decode_slowdown_vs_dynamic": (
                paged["decode_wall_s_median"] / max(1e-9, dyn["decode_wall_s_median"])
            ),
            "paged_prefill_slowdown_vs_dynamic": (
                paged["prefill_wall_s_median"] / max(1e-9, dyn["prefill_wall_s_median"])
            ),
            "paged_e2e_slowdown_vs_dynamic": (
                paged["total_wall_s_median"] / max(1e-9, dyn["total_wall_s_median"])
            ),
            "dynamic_prefill_fraction": dyn["prefill_fraction"],
            "paged_prefill_fraction": paged["prefill_fraction"],
        },
    }
    out = ROOT / args.output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["comparison"], indent=2))
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
