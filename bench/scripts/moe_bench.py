from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from runtime.phase2.moe_primitive import ToyMoE


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ToyMoE(dim=256, experts=4, top_k=2, capacity_factor=1.25).to(device).eval()
    reqs, seq_len = 20, 32
    x = torch.randn(reqs, seq_len, 256, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        y, stats = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-6)
    tokens = reqs * seq_len
    active_ratio = float(model.top_k / model.experts)
    out = {
        "tokens_per_sec": tokens / elapsed,
        "model_id": "toy_moe_4e_d256",
        "experts_per_token": stats["experts_per_token"],
        "total_experts": stats["total_experts"],
        "active_param_ratio": active_ratio,
        "routing": stats,
        "output_mean_abs": float(y.abs().mean().item()),
    }
    p = Path("bench/results/moe-synthetic.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(p)


if __name__ == "__main__":
    main()
