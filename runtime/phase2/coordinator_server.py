from __future__ import annotations

import asyncio
import collections
from contextlib import asynccontextmanager
import json
import os
import sqlite3
import time
import uuid
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
DURABLE_DB_PATH = os.getenv("COORDINATOR_DB_PATH", f"{LOG_DIR}/coordinator_state.db")


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
    tenant_id: str = "default"


class WorkerState(BaseModel):
    url: str
    role: str = "decode"
    healthy: bool = False
    inflight: int = 0
    last_ok_unix_ms: int = 0
    avg_latency_ms: float = 0.0
    completed: int = 0
    health_failures: int = 0


_workers: list[WorkerState] = []
_active: collections.OrderedDict[str, tuple[float, str]] = collections.OrderedDict()
_completed_cache: collections.OrderedDict[str, tuple[float, dict[str, Any]]] = collections.OrderedDict()
_cancelled: collections.OrderedDict[str, float] = collections.OrderedDict()
_pending_recovery: set[str] = set()
_health_task: asyncio.Task | None = None
_autoscale_task: asyncio.Task | None = None
_ttft_ms_samples: collections.deque[float] = collections.deque(maxlen=1000)
_queue_delay_ms_samples: collections.deque[float] = collections.deque(maxlen=1000)
_tenant_inflight: dict[str, int] = {}
_admission_limit = int(os.getenv("COORDINATOR_ADMISSION_MAX", "64"))
_admission_min = int(os.getenv("COORDINATOR_ADMISSION_MIN", "8"))
_admission_max = int(os.getenv("COORDINATOR_ADMISSION_MAX", "64"))
_autoscale_target_ttft_p95_ms = float(os.getenv("COORDINATOR_TARGET_TTFT_P95_MS", "300"))
_autoscale_target_inflight = int(os.getenv("COORDINATOR_TARGET_INFLIGHT", "16"))
_sched_policy = os.getenv("COORDINATOR_SCHED_POLICY", "latency").strip().lower()
_tenant_limits_raw = os.getenv("COORDINATOR_TENANT_LIMITS_JSON", '{"default":16}')
_tenant_weights_raw = os.getenv("COORDINATOR_TENANT_WEIGHTS_JSON", '{"default":1.0}')
_tenant_limits: dict[str, int] = {"default": 16}
_tenant_weights: dict[str, float] = {"default": 1.0}
_admission_wait_timeout_ms = int(os.getenv("COORDINATOR_ADMISSION_WAIT_TIMEOUT_MS", "30000"))
_default_deadline_ms = int(os.getenv("COORDINATOR_DEFAULT_DEADLINE_MS", "120000"))
_health_timeout_s = float(os.getenv("COORDINATOR_HEALTH_TIMEOUT_S", "5.0"))
_health_unhealthy_threshold = int(os.getenv("COORDINATOR_HEALTH_UNHEALTHY_THRESHOLD", "3"))
def _worker_adapter(model_field: str) -> str:
    base_model = os.getenv("HF_MODEL_ID", "").strip()
    known = {n.strip() for n in os.getenv("LORA_ADAPTER_NAMES", "base,adapter_a,adapter_b,adapter_c").split(",") if n.strip()}
    if model_field in known:
        return model_field
    if base_model and model_field and model_field != base_model:
        return model_field
    return "base"


_metrics: dict[str, int] = {
    "requests_total": 0,
    "stream_requests_total": 0,
    "nonstream_requests_total": 0,
    "retry_attempts_total": 0,
    "cancellations_total": 0,
    "tenant_rejections_total": 0,
    "admission_rejections_total": 0,
    "worker_transport_failures_total": 0,
    "worker_stream_transport_failures_total": 0,
    "request_timeouts_total": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _health_task, _autoscale_task
    worker_urls = os.getenv("WORKER_URLS", "http://127.0.0.1:8102")
    prefill_worker_urls = os.getenv("PREFILL_WORKER_URLS", "").strip()
    decode_worker_urls = os.getenv("DECODE_WORKER_URLS", "").strip()
    _workers.clear()
    _parse_tenant_controls()
    _ensure_durable_store()
    if prefill_worker_urls or decode_worker_urls:
        for raw in prefill_worker_urls.split(","):
            u = raw.strip()
            if u:
                _workers.append(WorkerState(url=u, role="prefill"))
        for raw in decode_worker_urls.split(","):
            u = raw.strip()
            if u:
                _workers.append(WorkerState(url=u, role="decode"))
    else:
        for raw in worker_urls.split(","):
            u = raw.strip()
            if u:
                _workers.append(WorkerState(url=u, role="decode"))
    _load_recovery_state()
    _health_task = asyncio.create_task(_health_loop())
    _autoscale_task = asyncio.create_task(_autoscale_loop())
    try:
        yield
    finally:
        if _health_task is not None:
            _health_task.cancel()
            try:
                await _health_task
            except asyncio.CancelledError:
                pass
        if _autoscale_task is not None:
            _autoscale_task.cancel()
            try:
                await _autoscale_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="phase2-coordinator", lifespan=lifespan)


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * min(1.0, max(0.0, q))
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _parse_tenant_controls() -> None:
    global _tenant_limits, _tenant_weights
    try:
        raw_limits = json.loads(_tenant_limits_raw)
        _tenant_limits = {str(k): int(v) for k, v in raw_limits.items()}
    except Exception:
        _tenant_limits = {"default": 16}
    try:
        raw_weights = json.loads(_tenant_weights_raw)
        _tenant_weights = {str(k): float(v) for k, v in raw_weights.items()}
    except Exception:
        _tenant_weights = {"default": 1.0}
    if "default" not in _tenant_limits:
        _tenant_limits["default"] = 16
    if "default" not in _tenant_weights:
        _tenant_weights["default"] = 1.0


def _ensure_durable_store() -> None:
    _ensure_log_dir()
    conn = sqlite3.connect(DURABLE_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS request_state (
                request_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_unix_ms INTEGER NOT NULL,
                response_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS request_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                event TEXT NOT NULL,
                ts_unix_ms INTEGER NOT NULL,
                payload_json TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _persist_event(request_id: str, tenant_id: str, event: str, payload: dict[str, Any] | None = None) -> None:
    _ensure_durable_store()
    now_ms = int(time.time() * 1000)
    response_json: str | None = None
    if payload and "response" in payload:
        response_json = json.dumps(payload["response"])
    conn = sqlite3.connect(DURABLE_DB_PATH)
    try:
        conn.execute(
            "INSERT INTO request_events (request_id, tenant_id, event, ts_unix_ms, payload_json) VALUES (?, ?, ?, ?, ?)",
            (request_id, tenant_id, event, now_ms, json.dumps(payload or {})),
        )
        conn.execute(
            """
            INSERT INTO request_state (request_id, tenant_id, status, updated_unix_ms, response_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                tenant_id=excluded.tenant_id,
                status=excluded.status,
                updated_unix_ms=excluded.updated_unix_ms,
                response_json=COALESCE(excluded.response_json, request_state.response_json)
            """,
            (request_id, tenant_id, event, now_ms, response_json),
        )
        conn.commit()
    finally:
        conn.close()


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
    _ensure_durable_store()
    conn = sqlite3.connect(DURABLE_DB_PATH)
    try:
        rows = conn.execute("SELECT request_id, status, response_json FROM request_state").fetchall()
    finally:
        conn.close()
    if rows:
        _pending_recovery.clear()
        for rid, status, response_json in rows:
            if status == "completed" and response_json:
                _cache_completion(rid, json.loads(response_json))
            elif status in {"admitted", "worker_stream_transport_failed", "worker_transport_failed"}:
                _pending_recovery.add(rid)
            elif status == "cancelled":
                _mark_cancelled(rid)
        return

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
        async with httpx.AsyncClient(timeout=_health_timeout_s) as client:
            for w in _workers:
                try:
                    r = await client.get(f"{w.url}/healthz")
                    w.healthy = r.status_code == 200
                    if w.healthy:
                        w.last_ok_unix_ms = int(time.time() * 1000)
                        w.health_failures = 0
                    else:
                        w.health_failures += 1
                        if w.health_failures >= _health_unhealthy_threshold:
                            w.healthy = False
                except Exception:
                    w.health_failures += 1
                    if w.health_failures >= _health_unhealthy_threshold:
                        w.healthy = False
        await asyncio.sleep(1.0)


def _choose_worker(exclude_urls: set[str] | None = None, role: str = "decode", tenant_key: str = "default") -> WorkerState:
    exclude_urls = exclude_urls or set()
    healthy = [w for w in _workers if w.healthy and w.url not in exclude_urls and w.role == role]
    if not healthy:
        raise HTTPException(status_code=503, detail="No healthy workers")
    if _sched_policy == "throughput":
        return min(healthy, key=lambda w: (w.inflight, -w.completed))
    if _sched_policy == "fairshare":
        tenant_load = _tenant_inflight.get(tenant_key, 0)
        tenant_weight = max(0.1, _tenant_weights.get(tenant_key, _tenant_weights["default"]))
        return min(healthy, key=lambda w: (((tenant_load + w.inflight) / tenant_weight), w.avg_latency_ms or 0.0))
    # default latency-oriented
    return min(healthy, key=lambda w: (w.avg_latency_ms or 0.0, w.inflight))


async def _admit_or_wait(tenant_key: str) -> None:
    start = time.perf_counter()
    timeout_s = max(0.001, _admission_wait_timeout_ms / 1000.0)
    while len(_active) >= _admission_limit:
        if (time.perf_counter() - start) >= timeout_s:
            _metrics["admission_rejections_total"] += 1
            raise HTTPException(status_code=429, detail="Coordinator admission queue timeout")
        await asyncio.sleep(0.005)
    tenant_limit = _tenant_limits.get(tenant_key, _tenant_limits["default"])
    if _tenant_inflight.get(tenant_key, 0) >= tenant_limit:
        _metrics["tenant_rejections_total"] += 1
        raise HTTPException(status_code=429, detail=f"Tenant inflight limit reached for tenant={tenant_key}")


async def _autoscale_loop() -> None:
    global _admission_limit
    while True:
        await asyncio.sleep(1.0)
        ttft_p95 = _pctl(list(_ttft_ms_samples), 0.95)
        inflight = len(_active)
        if ttft_p95 > _autoscale_target_ttft_p95_ms or inflight > _autoscale_target_inflight:
            _admission_limit = max(_admission_min, _admission_limit - 1)
        elif inflight < max(1, _autoscale_target_inflight // 2):
            _admission_limit = min(_admission_max, _admission_limit + 1)


async def _prefill_then_decode(
    req: ChatRequest, prompt: str, request_id: str, deadline_ms: int | None, tenant_key: str
) -> tuple[WorkerState, dict[str, Any]]:
    prefill_worker = _choose_worker(role="prefill", tenant_key=tenant_key)
    decode_worker = _choose_worker(role="decode", tenant_key=tenant_key)
    prefill_worker.inflight += 1
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            prefill = await client.post(
                f"{prefill_worker.url}/prefill",
                json={
                    "request_id": request_id,
                    "prompt": prompt,
                    "max_tokens": req.max_tokens,
                    "temperature": req.temperature,
                    "deadline_unix_ms": deadline_ms,
                },
            )
            prefill.raise_for_status()
            return decode_worker, prefill.json()
    finally:
        prefill_worker.inflight = max(0, prefill_worker.inflight - 1)


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
    tenant_key: str,
    prefill_handoff_id: str | None = None,
) -> Any:
    generated: list[str] = []
    first_token_ttft_ms: float | None = None
    stream_start = time.perf_counter()
    prompt_tokens = max(1, len(prompt.split()))
    try:
        timeout_s = float(os.getenv("COORDINATOR_WORKER_STREAM_TIMEOUT_S", "900"))
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
                    "prefill_handoff_id": prefill_handoff_id,
                    "adapter": _worker_adapter(req.model),
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
                        if first_token_ttft_ms is None:
                            first_token_ttft_ms = (time.perf_counter() - stream_start) * 1000.0
                            _ttft_ms_samples.append(first_token_ttft_ms)
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
                        _persist_event(request_id, tenant_key, "completed", {"response": result})
                        yield _stream_done(req.model, prompt_tokens, max(1, len(generated)))
                        yield "data: [DONE]\n\n"
                        break
                    elif kind == "timed_out":
                        _metrics["request_timeouts_total"] += 1
                        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
                        _persist_event(request_id, tenant_key, "timed_out")
                        break
                    elif kind == "cancelled":
                        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "cancelled", "request_id": request_id})
                        _persist_event(request_id, tenant_key, "cancelled")
                        break
    except httpx.TimeoutException:
        _metrics["request_timeouts_total"] += 1
        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
        _persist_event(request_id, tenant_key, "timed_out")
    except Exception:
        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "failed", "request_id": request_id})
        _persist_event(request_id, tenant_key, "failed")
        raise
    finally:
        _active.pop(request_id, None)
        worker.inflight = max(0, worker.inflight - 1)


async def _worker_stream_with_retries(
    req: ChatRequest, prompt: str, request_id: str, deadline_ms: int | None, tenant_key: str
) -> Any:
    skipped_workers: set[str] = set()
    emitted_tokens = False
    while True:
        worker = _choose_worker(skipped_workers, role="decode", tenant_key=tenant_key)
        _mark_active(request_id, worker.url)
        worker.inflight += 1
        _tenant_inflight[tenant_key] = _tenant_inflight.get(tenant_key, 0) + 1
        _append_log(
            {
                "ts_unix_ms": int(time.time() * 1000),
                "event": "admitted",
                "request_id": request_id,
                "worker_url": worker.url,
                "deadline_ms": deadline_ms,
            }
        )
        _persist_event(request_id, tenant_key, "admitted", {"worker_url": worker.url, "deadline_ms": deadline_ms})
        try:
            try:
                async for chunk in _worker_stream(req, prompt, request_id, worker, deadline_ms, tenant_key):
                    emitted_tokens = emitted_tokens or '"content"' in chunk
                    yield chunk
                return
            finally:
                _tenant_inflight[tenant_key] = max(0, _tenant_inflight.get(tenant_key, 1) - 1)
        except httpx.RequestError as exc:
            skipped_workers.add(worker.url)
            worker.healthy = False
            _metrics["retry_attempts_total"] += 1
            _metrics["worker_stream_transport_failures_total"] += 1
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
            _persist_event(
                request_id,
                tenant_key,
                "worker_stream_transport_failed",
                {"worker_url": worker.url, "error": str(exc), "emitted_tokens": emitted_tokens},
            )
            if emitted_tokens or len(skipped_workers) >= len(_workers):
                raise
        if deadline_ms is not None and deadline_ms <= int(time.time() * 1000):
            _metrics["request_timeouts_total"] += 1
            _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
            _persist_event(request_id, tenant_key, "timed_out")
            return


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(req: ChatRequest) -> dict[str, Any] | StreamingResponse:
    now_ms = int(time.time() * 1000)
    deadline_ms = req.deadline_ms if req.deadline_ms is not None else now_ms + _default_deadline_ms
    if deadline_ms is not None and deadline_ms <= now_ms:
        raise HTTPException(status_code=408, detail="Deadline already expired")

    prompt = "\n".join(m.content for m in req.messages)
    request_id = req.request_id or f"req-{uuid.uuid4().hex}"
    _metrics["requests_total"] += 1
    tenant_key = req.tenant_id or "default"
    await _admit_or_wait(tenant_key)

    # Replay-safe: return cached terminal completion for duplicate request id.
    cached = _get_completion(request_id)
    if cached is not None:
        return cached
    if _is_cancelled(request_id):
        raise HTTPException(status_code=499, detail="Request previously cancelled")

    if req.stream:
        _metrics["stream_requests_total"] += 1
        has_prefill = any(w.healthy and w.role == "prefill" for w in _workers)
        if has_prefill:
            decode_worker, handoff = await _prefill_then_decode(req, prompt, request_id, deadline_ms, tenant_key)
            _mark_active(request_id, decode_worker.url)
            decode_worker.inflight += 1
            _tenant_inflight[tenant_key] = _tenant_inflight.get(tenant_key, 0) + 1
            async def stream_with_handoff() -> Any:
                start = time.perf_counter()
                try:
                    async for chunk in _worker_stream(
                        req,
                        prompt,
                        request_id,
                        decode_worker,
                        deadline_ms,
                        tenant_key,
                        prefill_handoff_id=handoff.get("prefill_handoff_id"),
                    ):
                        yield chunk
                finally:
                    elapsed = (time.perf_counter() - start) * 1000.0
                    decode_worker.avg_latency_ms = elapsed if decode_worker.completed == 0 else (0.8 * decode_worker.avg_latency_ms + 0.2 * elapsed)
                    decode_worker.completed += 1
                    _tenant_inflight[tenant_key] = max(0, _tenant_inflight.get(tenant_key, 1) - 1)
            return StreamingResponse(stream_with_handoff(), media_type="text/event-stream")
        return StreamingResponse(
            _worker_stream_with_retries(req, prompt, request_id, deadline_ms, tenant_key),
            media_type="text/event-stream",
        )
    _metrics["nonstream_requests_total"] += 1

    skipped_workers: set[str] = set()
    while True:
        worker = _choose_worker(skipped_workers, role="decode", tenant_key=tenant_key)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                start_time = time.perf_counter()
                timeout_s = 30.0
                if deadline_ms is not None:
                    timeout_s = max(0.050, (deadline_ms - int(time.time() * 1000)) / 1000.0)

                has_prefill = any(w.healthy and w.role == "prefill" for w in _workers)
                prefill_handoff_id: str | None = None
                if has_prefill:
                    chosen_decode, handoff = await _prefill_then_decode(req, prompt, request_id, deadline_ms, tenant_key)
                    worker = chosen_decode
                    prefill_handoff_id = handoff.get("prefill_handoff_id")
                _mark_active(request_id, worker.url)
                worker.inflight += 1
                _tenant_inflight[tenant_key] = _tenant_inflight.get(tenant_key, 0) + 1
                _append_log(
                    {
                        "ts_unix_ms": int(time.time() * 1000),
                        "event": "admitted",
                        "request_id": request_id,
                        "worker_url": worker.url,
                        "deadline_ms": deadline_ms,
                    }
                )
                _persist_event(request_id, tenant_key, "admitted", {"worker_url": worker.url, "deadline_ms": deadline_ms})
                resp = await client.post(
                    f"{worker.url}/generate",
                    json={
                        "request_id": request_id,
                        "prompt": prompt,
                        "max_tokens": req.max_tokens,
                        "temperature": req.temperature,
                        "deadline_unix_ms": deadline_ms,
                        "prefill_handoff_id": prefill_handoff_id,
                        "adapter": _worker_adapter(req.model),
                    },
                    timeout=timeout_s,
                )
                resp.raise_for_status()
                payload = resp.json()
                elapsed = (time.perf_counter() - start_time) * 1000.0
                worker.avg_latency_ms = elapsed if worker.completed == 0 else (0.8 * worker.avg_latency_ms + 0.2 * elapsed)
                worker.completed += 1
                break
        except httpx.TimeoutException:
            _metrics["request_timeouts_total"] += 1
            _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
            _persist_event(request_id, tenant_key, "timed_out")
            raise HTTPException(status_code=408, detail="Deadline exceeded")
        except httpx.RequestError as exc:
            skipped_workers.add(worker.url)
            worker.healthy = False
            _metrics["worker_transport_failures_total"] += 1
            _append_log(
                {
                    "ts_unix_ms": int(time.time() * 1000),
                    "event": "worker_transport_failed",
                    "request_id": request_id,
                    "worker_url": worker.url,
                    "error": str(exc),
                }
            )
            _persist_event(request_id, tenant_key, "worker_transport_failed", {"worker_url": worker.url, "error": str(exc)})
            if len(skipped_workers) >= len(_workers):
                _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "failed", "request_id": request_id})
                _persist_event(request_id, tenant_key, "failed")
                raise HTTPException(status_code=503, detail="No reachable workers") from exc
        except Exception:
            _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "failed", "request_id": request_id})
            _persist_event(request_id, tenant_key, "failed")
            raise
        finally:
            _active.pop(request_id, None)
            worker.inflight = max(0, worker.inflight - 1)
            _tenant_inflight[tenant_key] = max(0, _tenant_inflight.get(tenant_key, 1) - 1)

        if deadline_ms is not None and deadline_ms <= int(time.time() * 1000):
            _metrics["request_timeouts_total"] += 1
            _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
            _persist_event(request_id, tenant_key, "timed_out")
            raise HTTPException(status_code=408, detail="Deadline exceeded")

    if payload.get("cancelled"):
        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "cancelled", "request_id": request_id})
        _persist_event(request_id, tenant_key, "cancelled")
        raise HTTPException(status_code=499, detail="Cancelled")
    if payload.get("timed_out"):
        _metrics["request_timeouts_total"] += 1
        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
        _persist_event(request_id, tenant_key, "timed_out")
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
    _persist_event(request_id, tenant_key, "completed", {"response": result})
    return result


@app.post("/v1/requests/{request_id}/cancel")
async def cancel_request(request_id: str) -> dict[str, str]:
    _metrics["cancellations_total"] += 1
    _mark_cancelled(request_id)
    worker_url = _get_active_worker(request_id)
    if worker_url:
        async with httpx.AsyncClient(timeout=3.0) as client:
            try:
                await client.post(f"{worker_url}/cancel/{request_id}")
            except Exception:
                pass
    _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "cancelled", "request_id": request_id})
    _persist_event(request_id, "default", "cancelled")
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
        "durable_db_path": DURABLE_DB_PATH,
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


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    ttft_values = list(_ttft_ms_samples)
    return {
        "requests_total": _metrics["requests_total"],
        "stream_requests_total": _metrics["stream_requests_total"],
        "nonstream_requests_total": _metrics["nonstream_requests_total"],
        "retry_attempts_total": _metrics["retry_attempts_total"],
        "cancellations_total": _metrics["cancellations_total"],
        "tenant_rejections_total": _metrics["tenant_rejections_total"],
        "admission_rejections_total": _metrics["admission_rejections_total"],
        "worker_transport_failures_total": _metrics["worker_transport_failures_total"],
        "worker_stream_transport_failures_total": _metrics["worker_stream_transport_failures_total"],
        "request_timeouts_total": _metrics["request_timeouts_total"],
        "current_inflight_requests": len(_active),
        "workers_inflight_total": sum(w.inflight for w in _workers),
        "ttft_ms_p50": _pctl(ttft_values, 0.50),
        "ttft_ms_p95": _pctl(ttft_values, 0.95),
        "ttft_sample_count": len(ttft_values),
        "scheduler_policy": _sched_policy,
        "autoscale_admission_limit": _admission_limit,
        "autoscale_target_ttft_p95_ms": _autoscale_target_ttft_p95_ms,
        "autoscale_target_inflight": _autoscale_target_inflight,
        "tenant_limits": _tenant_limits,
        "tenant_weights": _tenant_weights,
    }
