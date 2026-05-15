from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

WORKER_URL = "http://127.0.0.1:8102"
app = FastAPI(title="phase2-coordinator")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = 0.0
    stream: bool = False


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> dict[str, Any]:
    prompt = "\n".join(m.content for m in req.messages)
    request_id = f"req-{int(time.time()*1e6)}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{WORKER_URL}/generate",
            json={
                "request_id": request_id,
                "prompt": prompt,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
            },
        )
        resp.raise_for_status()
        payload = resp.json()

    text = payload["text"]
    return {
        "id": "chatcmpl-phase2",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": max(1, len(prompt.split())),
            "completion_tokens": max(1, len(text.split())),
            "total_tokens": max(1, len(prompt.split())) + max(1, len(text.split())),
        },
    }
