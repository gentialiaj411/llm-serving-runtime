from __future__ import annotations

import asyncio
import collections
from contextlib import asynccontextmanager
import json
import os
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

LOG_DIR = "runtime/logs"
REQUEST_LOG = f"{LOG_DIR}/requests.jsonl"
REQUEST_LOG_MAX_BYTES = int(os.getenv("COORDINATOR_REQUEST_LOG_MAX_BYTES", str(16 * 1024 * 1024)))
CACHE_TTL_S = float(os.getenv("COORDINATOR_CACHE_TTL_S", "3600"))
ACTIVE_CACHE_MAX = int(os.getenv("COORDINATOR_ACTIVE_CACHE_MAX", "10000"))
COMPLETED_CACHE_MAX = int(os.getenv("COORDINATOR_COMPLETED_CACHE_MAX", "10000"))
CANCELLED_CACHE_MAX = int(os.getenv("COORDINATOR_CANCELLED_CACHE_MAX", "10000"))


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
_active: collections.OrderedDict[str, tuple[float, str]] = collections.OrderedDict()
_completed_cache: collections.OrderedDict[str, tuple[float, dict[str, Any]]] = collections.OrderedDict()
_cancelled: collections.OrderedDict[str, float] = collections.OrderedDict()
_pending_recovery: set[str] = set()
_health_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _health_task
    worker_urls = os.getenv("WORKER_URLS", "http://127.0.0.1:8102")
    _workers.clear()
    for raw in worker_urls.split(","):
        u = raw.strip()
        if u:
            _workers.append(WorkerState(url=u))
    _load_recovery_state()
    _health_task = asyncio.create_task(_health_loop())
    try:
        yield
    finally:
        if _health_task is not None:
            _health_task.cancel()
            try:
                await _health_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="phase2-coordinator", lifespan=lifespan)


def _prune_ordered_cache(cache: collections.OrderedDict, max_items: int, now: float | None = None) -> None:
    now = now or time.time()
    expired = [key for key, value in cache.items() if now - (value[0] if isinstance(value, tuple) else value) > CACHE_TTL_S]
    for key in expired:
        cache.pop(key, None)
    while len(cache) > max_items:
        cache.popitem(last=False)


def _cache_completion(request_id: str, response: dict[str, Any]) -> None:
    _completed_cache[request_id] = (time.time(), response)
    _completed_cache.move_to_end(request_id)
    _prune_ordered_cache(_completed_cache, COMPLETED_CACHE_MAX)


def _get_completion(request_id: str) -> dict[str, Any] | None:
    _prune_ordered_cache(_completed_cache, COMPLETED_CACHE_MAX)
    cached = _completed_cache.get(request_id)
    if cached is None:
        return None
    _completed_cache.move_to_end(request_id)
    return cached[1]


def _mark_cancelled(request_id: str) -> None:
    _cancelled[request_id] = time.time()
    _cancelled.move_to_end(request_id)
    _prune_ordered_cache(_cancelled, CANCELLED_CACHE_MAX)


def _is_cancelled(request_id: str) -> bool:
    _prune_ordered_cache(_cancelled, CANCELLED_CACHE_MAX)
    return request_id in _cancelled


def _mark_active(request_id: str, worker_url: str) -> None:
    _active[request_id] = (time.time(), worker_url)
    _active.move_to_end(request_id)
    _prune_ordered_cache(_active, ACTIVE_CACHE_MAX)


def _get_active_worker(request_id: str) -> str | None:
    _prune_ordered_cache(_active, ACTIVE_CACHE_MAX)
    active = _active.get(request_id)
    if active is None:
        return None
    _active.move_to_end(request_id)
    return active[1]


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _append_log(event: dict[str, Any]) -> None:
    _ensure_log_dir()
    if REQUEST_LOG_MAX_BYTES > 0 and os.path.exists(REQUEST_LOG) and os.path.getsize(REQUEST_LOG) > REQUEST_LOG_MAX_BYTES:
        rotated = f"{REQUEST_LOG}.1"
        if os.path.exists(rotated):
            os.remove(rotated)
        os.replace(REQUEST_LOG, rotated)
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
                    _cache_completion(rid, ev["response"])
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


def _choose_worker(exclude_urls: set[str] | None = None) -> WorkerState:
    exclude_urls = exclude_urls or set()
    healthy = [w for w in _workers if w.healthy and w.url not in exclude_urls]
    if not healthy:
        raise HTTPException(status_code=503, detail="No healthy workers")
    return min(healthy, key=lambda w: w.inflight)


def _stream_chunk(model: str, content: str) -> str:
    payload = {
        "id": "chatcmpl-phase2",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _stream_done(model: str, prompt_tokens: int, completion_tokens: int) -> str:
    payload = {
        "id": "chatcmpl-phase2",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _worker_stream(
    req: ChatRequest,
    prompt: str,
    request_id: str,
    worker: WorkerState,
    deadline_ms: int | None,
) -> Any:
    generated: list[str] = []
    prompt_tokens = max(1, len(prompt.split()))
    try:
        timeout_s = 30.0
        if deadline_ms is not None:
            timeout_s = max(0.050, (deadline_ms - int(time.time() * 1000)) / 1000.0)

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{worker.url}/generate_stream",
                json={
                    "request_id": request_id,
                    "prompt": prompt,
                    "max_tokens": req.max_tokens,
                    "temperature": req.temperature,
                    "deadline_unix_ms": deadline_ms,
                },
                timeout=timeout_s,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    kind = event.get("type")
                    if kind == "token":
                        token = str(event.get("token", ""))
                        generated.append(token)
                        content = token if len(generated) == 1 else f" {token}"
                        yield _stream_chunk(req.model, content)
                    elif kind == "done":
                        text = " ".join(generated)
                        result = {
                            "id": "chatcmpl-phase2",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": max(1, len(generated)),
                                "total_tokens": prompt_tokens + max(1, len(generated)),
                            },
                        }
                        _cache_completion(request_id, result)
                        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "completed", "request_id": request_id, "response": result})
                        yield _stream_done(req.model, prompt_tokens, max(1, len(generated)))
                        yield "data: [DONE]\n\n"
                        break
                    elif kind == "timed_out":
                        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
                        break
                    elif kind == "cancelled":
                        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "cancelled", "request_id": request_id})
                        break
    except httpx.TimeoutException:
        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
    except Exception:
        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "failed", "request_id": request_id})
        raise
    finally:
        _active.pop(request_id, None)
        worker.inflight = max(0, worker.inflight - 1)


async def _worker_stream_with_retries(req: ChatRequest, prompt: str, request_id: str, deadline_ms: int | None) -> Any:
    skipped_workers: set[str] = set()
    emitted_tokens = False
    while True:
        worker = _choose_worker(skipped_workers)
        _mark_active(request_id, worker.url)
        worker.inflight += 1
        _append_log(
            {
                "ts_unix_ms": int(time.time() * 1000),
                "event": "admitted",
                "request_id": request_id,
                "worker_url": worker.url,
                "deadline_ms": deadline_ms,
            }
        )
        try:
            async for chunk in _worker_stream(req, prompt, request_id, worker, deadline_ms):
                emitted_tokens = emitted_tokens or '"content"' in chunk
                yield chunk
            return
        except httpx.RequestError as exc:
            worker.healthy = False
            skipped_workers.add(worker.url)
            _append_log(
                {
                    "ts_unix_ms": int(time.time() * 1000),
                    "event": "worker_stream_transport_failed",
                    "request_id": request_id,
                    "worker_url": worker.url,
                    "error": str(exc),
                    "emitted_tokens": emitted_tokens,
                }
            )
            if emitted_tokens or len(skipped_workers) >= len(_workers):
                raise
        if deadline_ms is not None and deadline_ms <= int(time.time() * 1000):
            _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
            return


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(req: ChatRequest) -> dict[str, Any] | StreamingResponse:
    now_ms = int(time.time() * 1000)
    deadline_ms = req.deadline_ms
    if deadline_ms is not None and deadline_ms <= now_ms:
        raise HTTPException(status_code=408, detail="Deadline already expired")

    prompt = "\n".join(m.content for m in req.messages)
    request_id = req.request_id or f"req-{int(time.time()*1e6)}"

    # Replay-safe: return cached terminal completion for duplicate request id.
    cached = _get_completion(request_id)
    if cached is not None:
        return cached
    if _is_cancelled(request_id):
        raise HTTPException(status_code=499, detail="Request previously cancelled")

    if req.stream:
        return StreamingResponse(
            _worker_stream_with_retries(req, prompt, request_id, deadline_ms),
            media_type="text/event-stream",
        )

    skipped_workers: set[str] = set()
    while True:
        worker = _choose_worker(skipped_workers)
        _mark_active(request_id, worker.url)
        worker.inflight += 1
        _append_log(
            {
                "ts_unix_ms": int(time.time() * 1000),
                "event": "admitted",
                "request_id": request_id,
                "worker_url": worker.url,
                "deadline_ms": deadline_ms,
            }
        )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                break
        except httpx.TimeoutException:
            _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
            raise HTTPException(status_code=408, detail="Deadline exceeded")
        except httpx.RequestError as exc:
            worker.healthy = False
            skipped_workers.add(worker.url)
            _append_log(
                {
                    "ts_unix_ms": int(time.time() * 1000),
                    "event": "worker_transport_failed",
                    "request_id": request_id,
                    "worker_url": worker.url,
                    "error": str(exc),
                }
            )
            if len(skipped_workers) >= len(_workers):
                _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "failed", "request_id": request_id})
                raise HTTPException(status_code=503, detail="No reachable workers") from exc
        except Exception:
            _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "failed", "request_id": request_id})
            raise
        finally:
            _active.pop(request_id, None)
            worker.inflight = max(0, worker.inflight - 1)

        if deadline_ms is not None and deadline_ms <= int(time.time() * 1000):
            _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
            raise HTTPException(status_code=408, detail="Deadline exceeded")

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
    _cache_completion(request_id, result)
    _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "completed", "request_id": request_id, "response": result})
    return result


@app.post("/v1/requests/{request_id}/cancel")
async def cancel_request(request_id: str) -> dict[str, str]:
    _mark_cancelled(request_id)
    worker_url = _get_active_worker(request_id)
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
