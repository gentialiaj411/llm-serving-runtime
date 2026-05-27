from __future__ import annotations

import torch
import torch.nn as nn


class ToyMoE(nn.Module):
    def __init__(self, dim: int = 256, experts: int = 4, top_k: int = 2, capacity_factor: float = 1.25) -> None:
        super().__init__()
        self.dim = dim
        self.experts = experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(dim, experts, bias=False)
        self.ffns = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)) for _ in range(experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        b, t, d = x.shape
        tokens = b * t
        flat = x.reshape(tokens, d)
        scores = torch.softmax(self.router(flat), dim=-1)
        probs, idx = torch.topk(scores, self.top_k, dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        cap = max(1, int(self.capacity_factor * tokens * self.top_k / self.experts))
        out = torch.zeros_like(flat)
        dropped = 0
        counts = torch.zeros(self.experts, device=x.device, dtype=torch.int32)

        for expert in range(self.experts):
            mask = idx == expert
            token_ids = torch.nonzero(mask.any(dim=-1), as_tuple=False).squeeze(-1)
            if token_ids.numel() == 0:
                continue
            if token_ids.numel() > cap:
                dropped += int(token_ids.numel() - cap)
                token_ids = token_ids[:cap]
            counts[expert] = int(token_ids.numel())
            expert_in = flat[token_ids]
            expert_out = self.ffns[expert](expert_in)
            weights = torch.zeros(token_ids.shape[0], device=x.device, dtype=flat.dtype)
            for k in range(self.top_k):
                chosen = idx[token_ids, k] == expert
                weights = weights + probs[token_ids, k] * chosen.to(flat.dtype)
            out[token_ids] += expert_out * weights.unsqueeze(-1)

        stats = {
            "experts_per_token": self.top_k,
            "total_experts": self.experts,
            "capacity_per_expert": cap,
            "dropped_assignments": dropped,
            "mean_tokens_per_expert": float(counts.float().mean().item()),
        }
        return out.reshape(b, t, d), stats
