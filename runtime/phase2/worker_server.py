from __future__ import annotations

import asyncio
import collections
import time
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="phase2-worker")


class GenerateRequest(BaseModel):
    request_id: str
    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = 0.0
    deadline_unix_ms: int | None = None


@dataclass
class Pending:
    req: GenerateRequest
    fut: asyncio.Future


@dataclass
class ActiveState:
    req: GenerateRequest
    fut: asyncio.Future
    words: list[str]
    generated: list[str]
    cursor: int


_queue: asyncio.Queue[Pending] = asyncio.Queue()
_cancelled: set[str] = set()
_active: dict[str, ActiveState] = {}
_waiting: collections.deque[Pending] = collections.deque()


async def _continuous_batch_loop() -> None:
    # Orca-style idea: schedule at iteration boundaries, admitting new requests continuously.
    max_active = 32
    decode_step_ms = 2
    while True:
        # Pull at least one request if system is idle.
        if not _waiting and not _active:
            _waiting.append(await _queue.get())

        # Non-blocking drain of new arrivals.
        while True:
            try:
                _waiting.append(_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Admit waiting requests into active decode set.
        while _waiting and len(_active) < max_active:
            pending = _waiting.popleft()
            words = pending.req.prompt.split() or ["hello"]
            _active[pending.req.request_id] = ActiveState(
                req=pending.req,
                fut=pending.fut,
                words=words,
                generated=[],
                cursor=0,
            )

        if not _active:
            await asyncio.sleep(0.001)
            continue

        # One decode iteration: advance every active request by one token.
        await asyncio.sleep(decode_step_ms / 1000.0)
        now_ms = int(time.time() * 1000)
        finished_ids: list[str] = []

        for rid, state in list(_active.items()):
            if rid in _cancelled:
                if not state.fut.done():
                    state.fut.set_result({"request_id": rid, "text": "", "cancelled": True})
                finished_ids.append(rid)
                continue

            if state.req.deadline_unix_ms is not None and now_ms > state.req.deadline_unix_ms:
                if not state.fut.done():
                    state.fut.set_result({"request_id": rid, "text": "", "timed_out": True})
                finished_ids.append(rid)
                continue

            token = state.words[state.cursor % len(state.words)]
            state.generated.append(token)
            state.cursor += 1

            if len(state.generated) >= state.req.max_tokens:
                text = " ".join(state.generated)
                if not state.fut.done():
                    state.fut.set_result({"request_id": rid, "text": text, "cancelled": False})
                finished_ids.append(rid)

        for rid in finished_ids:
            _active.pop(rid, None)
            _cancelled.discard(rid)


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(_continuous_batch_loop())


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    await _queue.put(Pending(req=req, fut=fut))
    return await fut


@app.post("/cancel/{request_id}")
async def cancel(request_id: str) -> dict[str, str]:
    _cancelled.add(request_id)
    return {"request_id": request_id, "status": "cancel_accepted"}


@app.get("/metrics")
async def metrics() -> dict[str, int]:
    return {
        "queue_waiting": len(_waiting),
        "active_decode": len(_active),
    }
