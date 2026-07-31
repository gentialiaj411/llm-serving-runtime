from __future__ import annotations

import json
import os
import time
import gc
from pathlib import Path

import torch

from runtime.phase2.worker_server import ActiveState, GenerateRequest, TransformersBackend


def _run(mode: str) -> dict:
    os.environ["PHASE2_BACKEND"] = "transformers"
    os.environ["PHASE2_QUANT"] = mode
    os.environ["HF_DEVICE"] = "cuda"
    os.environ["HF_TORCH_DTYPE"] = "float16"
    if mode == "none":
        os.environ["HF_MODEL_ID"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    else:
        os.environ["HF_AWQ_MODEL_ID"] = os.getenv(
            "HF_AWQ_MODEL_ID",
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ",
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    backend = TransformersBackend()
    reqs = 50
    max_tokens = 64
    prompt = " ".join(["tok"] * 128)
    states = []
    for i in range(reqs):
        state = ActiveState(
            req=GenerateRequest(request_id=f"{mode}-{i}", prompt=prompt, max_tokens=max_tokens),
            fut=None,  # type: ignore[arg-type]
            stream_queue=None,
            words=[],
            generated=[],
            cursor=0,
        )
        backend.init_state(state)
        states.append(state)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max_tokens):
        backend.next_token_batch(states)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-6)
    total_tokens = reqs * max_tokens
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    row = {
        "mode": mode,
        "model_id": backend.model_id,
        "request_count": reqs,
        "prompt_len": 128,
        "max_tokens": max_tokens,
        "tokens_per_sec": total_tokens / elapsed,
        "peak_vram_bytes": peak,
    }
    del states
    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> None:
    rows = [_run("none"), _run("int4")]
    baseline = next(row for row in rows if row["mode"] == "none")
    awq = next(row for row in rows if row["mode"] == "int4")
    awq["vram_reduction_percent_vs_fp16"] = (
        100.0 * (baseline["peak_vram_bytes"] - awq["peak_vram_bytes"]) / baseline["peak_vram_bytes"]
        if baseline["peak_vram_bytes"]
        else 0.0
    )
    out = Path("bench/results/quantization-comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
