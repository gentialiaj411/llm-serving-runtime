from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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


def _stream_chunk(model: str, content: str) -> str:
    payload = {
        "id": "chatcmpl-phase1",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_completion(req: ChatRequest, prompt: str) -> Any:
    await asyncio.sleep(float(os.getenv("PHASE1_SIMULATED_PREFILL_MS", "20")) / 1000.0)
    decode_ms = float(os.getenv("PHASE1_SIMULATED_DECODE_MS", "2"))
    words = prompt.split() or ["hello"]
    for i in range(req.max_tokens):
        token = words[i % len(words)]
        content = token if i == 0 else f" {token}"
        yield _stream_chunk(req.model, content)
        await asyncio.sleep(decode_ms / 1000.0)

    done = {
        "id": "chatcmpl-phase1",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": max(1, len(prompt.split())),
            "completion_tokens": req.max_tokens,
            "total_tokens": max(1, len(prompt.split())) + req.max_tokens,
        },
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> dict[str, Any]:
    t0 = time.perf_counter()
    prompt = "\n".join([m.content for m in req.messages])

    if req.stream:
        return StreamingResponse(_stream_completion(req, prompt), media_type="text/event-stream")

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
