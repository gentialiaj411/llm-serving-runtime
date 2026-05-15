from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="phase2-coordinator")
LOG_DIR = "runtime/logs"
REQUEST_LOG = f"{LOG_DIR}/requests.jsonl"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = 0.0
    stream: bool = False
    request_id: str | None = None
    deadline_ms: int | None = None


class WorkerState(BaseModel):
    url: str
    healthy: bool = False
    inflight: int = 0
    last_ok_unix_ms: int = 0


_workers: list[WorkerState] = []
_active: dict[str, str] = {}
_completed_cache: dict[str, dict[str, Any]] = {}
_cancelled: set[str] = set()
_pending_recovery: set[str] = set()


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _append_log(event: dict[str, Any]) -> None:
    _ensure_log_dir()
    with open(REQUEST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _load_recovery_state() -> None:
    if not os.path.exists(REQUEST_LOG):
        return
    admitted: set[str] = set()
    terminal: set[str] = set()
    with open(REQUEST_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            rid = ev.get("request_id")
            kind = ev.get("event")
            if not rid:
                continue
            if kind == "admitted":
                admitted.add(rid)
            if kind in {"completed", "failed", "cancelled", "timed_out"}:
                terminal.add(rid)
                if kind == "completed" and "response" in ev:
                    _completed_cache[rid] = ev["response"]
    _pending_recovery.clear()
    _pending_recovery.update(admitted - terminal)


async def _health_loop() -> None:
    while True:
        async with httpx.AsyncClient(timeout=2.0) as client:
            for w in _workers:
                try:
                    r = await client.get(f"{w.url}/healthz")
                    w.healthy = r.status_code == 200
                    if w.healthy:
                        w.last_ok_unix_ms = int(time.time() * 1000)
                except Exception:
                    w.healthy = False
        await asyncio.sleep(1.0)


def _choose_worker() -> WorkerState:
    healthy = [w for w in _workers if w.healthy]
    if not healthy:
        raise HTTPException(status_code=503, detail="No healthy workers")
    return min(healthy, key=lambda w: w.inflight)


@app.on_event("startup")
async def startup() -> None:
    worker_urls = os.getenv("WORKER_URLS", "http://127.0.0.1:8102")
    _workers.clear()
    for raw in worker_urls.split(","):
        u = raw.strip()
        if u:
            _workers.append(WorkerState(url=u))
    _load_recovery_state()
    asyncio.create_task(_health_loop())


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    deadline_ms = req.deadline_ms
    if deadline_ms is not None and deadline_ms <= now_ms:
        raise HTTPException(status_code=408, detail="Deadline already expired")

    prompt = "\n".join(m.content for m in req.messages)
    request_id = req.request_id or f"req-{int(time.time()*1e6)}"

    # Replay-safe: return cached terminal completion for duplicate request id.
    if request_id in _completed_cache:
        return _completed_cache[request_id]
    if request_id in _cancelled:
        raise HTTPException(status_code=499, detail="Request previously cancelled")

    worker = _choose_worker()
    _active[request_id] = worker.url
    worker.inflight += 1
    _append_log(
        {
            "ts_unix_ms": now_ms,
            "event": "admitted",
            "request_id": request_id,
            "worker_url": worker.url,
            "deadline_ms": deadline_ms,
        }
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            timeout_s = 30.0
            if deadline_ms is not None:
                timeout_s = max(0.050, (deadline_ms - int(time.time() * 1000)) / 1000.0)

            resp = await client.post(
                f"{worker.url}/generate",
                json={
                    "request_id": request_id,
                    "prompt": prompt,
                    "max_tokens": req.max_tokens,
                    "temperature": req.temperature,
                    "deadline_unix_ms": deadline_ms,
                },
                timeout=timeout_s,
            )
            resp.raise_for_status()
            payload = resp.json()
        except httpx.TimeoutException:
            _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
            raise HTTPException(status_code=408, detail="Deadline exceeded")
        except Exception:
            _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "failed", "request_id": request_id})
            raise
        finally:
            _active.pop(request_id, None)
            worker.inflight = max(0, worker.inflight - 1)

    if payload.get("cancelled"):
        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "cancelled", "request_id": request_id})
        raise HTTPException(status_code=499, detail="Cancelled")
    if payload.get("timed_out"):
        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
        raise HTTPException(status_code=408, detail="Deadline exceeded")

    text = payload.get("text", "")
    result = {
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
    _completed_cache[request_id] = result
    _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "completed", "request_id": request_id, "response": result})
    return result


@app.post("/v1/requests/{request_id}/cancel")
async def cancel_request(request_id: str) -> dict[str, str]:
    _cancelled.add(request_id)
    worker_url = _active.get(request_id)
    if worker_url:
        async with httpx.AsyncClient(timeout=3.0) as client:
            try:
                await client.post(f"{worker_url}/cancel/{request_id}")
            except Exception:
                pass
    _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "cancelled", "request_id": request_id})
    return {"request_id": request_id, "status": "cancel_accepted"}


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    healthy = sum(1 for w in _workers if w.healthy)
    return {"status": "ok", "workers_healthy": healthy, "workers_total": len(_workers)}


@app.get("/admin/recovery")
async def recovery_state() -> dict[str, Any]:
    return {
        "pending_count": len(_pending_recovery),
        "pending_request_ids": sorted(_pending_recovery),
        "log_path": REQUEST_LOG,
    }


@app.get("/admin/kv-metrics")
async def kv_metrics() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=3.0) as client:
        for w in _workers:
            entry: dict[str, Any] = {"worker_url": w.url, "healthy": w.healthy}
            try:
                r = await client.get(f"{w.url}/metrics")
                if r.status_code == 200:
                    entry["metrics"] = r.json()
                else:
                    entry["error"] = f"status_{r.status_code}"
            except Exception as e:
                entry["error"] = str(e)
            results.append(entry)
    return {"workers": results}
