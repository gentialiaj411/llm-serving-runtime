from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="phase1-openai-shim")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = 0.0
    stream: bool = False


def _deterministic_generate(prompt: str, max_tokens: int) -> str:
    words = prompt.split()
    if not words:
        words = ["hello"]
    out = []
    for i in range(max_tokens):
        out.append(words[i % len(words)])
    return " ".join(out)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> dict[str, Any]:
    t0 = time.perf_counter()
    prompt = "\n".join([m.content for m in req.messages])

    # Phase 1 baseline: request-by-request execution, no batching.
    await asyncio.sleep(float(os.getenv("PHASE1_SIMULATED_PREFILL_MS", "20")) / 1000.0)
    text = _deterministic_generate(prompt, req.max_tokens)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "id": "chatcmpl-phase1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": max(1, len(prompt.split())),
            "completion_tokens": max(1, len(text.split())),
            "total_tokens": max(1, len(prompt.split())) + max(1, len(text.split())),
        },
        "_debug": {"latency_ms": latency_ms},
    }
