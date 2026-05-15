from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="phase2-worker")


class GenerateRequest(BaseModel):
    request_id: str
    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = 0.0


@dataclass
class Pending:
    req: GenerateRequest
    fut: asyncio.Future


_queue: asyncio.Queue[Pending] = asyncio.Queue()


def _deterministic_generate(prompt: str, max_tokens: int) -> str:
    words = prompt.split() or ["hello"]
    return " ".join(words[i % len(words)] for i in range(max_tokens))


async def _batch_loop() -> None:
    max_batch = 8
    batch_wait_ms = 8
    while True:
        first = await _queue.get()
        items = [first]
        t_deadline = time.perf_counter() + (batch_wait_ms / 1000.0)

        while len(items) < max_batch and time.perf_counter() < t_deadline:
            try:
                items.append(_queue.get_nowait())
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)

        # Naive batching: process together but no fused kernels.
        await asyncio.sleep(0.010)
        for item in items:
            text = _deterministic_generate(item.req.prompt, item.req.max_tokens)
            item.fut.set_result({"request_id": item.req.request_id, "text": text})


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(_batch_loop())


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, str]:
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    await _queue.put(Pending(req=req, fut=fut))
    return await fut
