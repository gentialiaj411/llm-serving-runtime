from __future__ import annotations

import asyncio
import collections
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
import time
from typing import Any, AsyncGenerator, AsyncIterator, TypedDict
import uuid

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

class _JsonFormatter(logging.Formatter):
    _SKIP = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        out: dict[str, Any] = {
            "ts_unix_ms": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k not in self._SKIP:
                out[k] = v
        return json.dumps(out, default=str)


def _configure_logging() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        root.addHandler(handler)
    root.setLevel(level)


_log = logging.getLogger("coordinator")


def _env_int(name: str, default: int, *, min_val: int | None = None, max_val: int | None = None) -> int:
    raw = os.getenv(name, str(default))
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(f"Env var {name}={raw!r}: expected integer") from None
    if min_val is not None and val < min_val:
        raise ValueError(f"Env var {name}={val}: must be >= {min_val}")
    if max_val is not None and val > max_val:
        raise ValueError(f"Env var {name}={val}: must be <= {max_val}")
    return val


def _env_float(name: str, default: float, *, min_val: float | None = None, max_val: float | None = None) -> float:
    raw = os.getenv(name, str(default))
    try:
        val = float(raw)
    except ValueError:
        raise ValueError(f"Env var {name}={raw!r}: expected float") from None
    if min_val is not None and val < min_val:
        raise ValueError(f"Env var {name}={val}: must be >= {min_val}")
    if max_val is not None and val > max_val:
        raise ValueError(f"Env var {name}={val}: must be <= {max_val}")
    return val


class _UsageDict(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _MessageDict(TypedDict):
    role: str
    content: str


class _ChoiceDict(TypedDict):
    index: int
    message: _MessageDict
    finish_reason: str


class CompletionResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[_ChoiceDict]
    usage: _UsageDict


LOG_DIR = "runtime/logs"
REQUEST_LOG = f"{LOG_DIR}/requests.jsonl"
REQUEST_LOG_MAX_BYTES = _env_int("COORDINATOR_REQUEST_LOG_MAX_BYTES", 16 * 1024 * 1024, min_val=0)
CACHE_TTL_S = _env_float("COORDINATOR_CACHE_TTL_S", 3600.0, min_val=1.0)
ACTIVE_CACHE_MAX = _env_int("COORDINATOR_ACTIVE_CACHE_MAX", 10000, min_val=1)
COMPLETED_CACHE_MAX = _env_int("COORDINATOR_COMPLETED_CACHE_MAX", 10000, min_val=1)
CANCELLED_CACHE_MAX = _env_int("COORDINATOR_CANCELLED_CACHE_MAX", 10000, min_val=1)
MAX_PROMPT_CHARS = _env_int("COORDINATOR_MAX_PROMPT_CHARS", 100_000, min_val=1)
# Fix 7: Circuit breaker thresholds for worker health.
# A worker is marked unhealthy after FAILURE_THRESHOLD consecutive health-check
# failures; it re-enters the pool only after RECOVERY_THRESHOLD consecutive
# successes.  Transport errors in the request path immediately flip healthy=False
# and also reset the success streak.
WORKER_FAILURE_THRESHOLD = _env_int("COORDINATOR_HEALTH_FAILURE_THRESHOLD", 2, min_val=1)
WORKER_RECOVERY_THRESHOLD = _env_int("COORDINATOR_HEALTH_RECOVERY_THRESHOLD", 3, min_val=1)
# Fix 8: Interval for the background cache-pruning task.
CACHE_PRUNE_INTERVAL_S = _env_float("COORDINATOR_CACHE_PRUNE_INTERVAL_S", 60.0, min_val=1.0)


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
    # Fix 7: streak counters for circuit-breaker hysteresis.
    consecutive_failures: int = 0
    consecutive_successes: int = 0


_workers: list[WorkerState] = []
_active: collections.OrderedDict[str, tuple[float, str]] = collections.OrderedDict()
_completed_cache: collections.OrderedDict[str, tuple[float, CompletionResponse]] = collections.OrderedDict()
_cancelled: collections.OrderedDict[str, float] = collections.OrderedDict()
_request_fingerprints: collections.OrderedDict[str, str] = collections.OrderedDict()
_pending_recovery: set[str] = set()
_health_task: asyncio.Task | None = None
_prune_task: asyncio.Task | None = None
# Fix 6: async log writer — queue populated by _append_log, drained by background task.
_log_queue: asyncio.Queue[dict[str, Any] | None] | None = None
_log_writer_task: asyncio.Task | None = None
_ttft_ms_samples: collections.deque[float] = collections.deque(maxlen=1000)
_metrics: dict[str, int] = {
    "requests_total": 0,
    "stream_requests_total": 0,
    "nonstream_requests_total": 0,
    "retry_attempts_total": 0,
    "cancellations_total": 0,
}
# Protects the check-then-dispatch sequence so two concurrent requests sharing
# the same request_id cannot both miss the cache and double-dispatch.
_admission_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _health_task, _prune_task, _log_queue, _log_writer_task
    worker_urls = os.getenv("WORKER_URLS", "http://127.0.0.1:8102")
    _configure_logging()
    _workers.clear()
    for raw in worker_urls.split(","):
        u = raw.strip()
        if u:
            _workers.append(WorkerState(url=u))
    _log.info("coordinator starting", extra={"worker_count": len(_workers)})
    _load_recovery_state()
    _log_queue = asyncio.Queue()
    _log_writer_task = asyncio.create_task(_log_writer_loop())
    _health_task = asyncio.create_task(_health_loop())
    _prune_task = asyncio.create_task(_prune_loop())
    try:
        yield
    finally:
        for task in (_health_task, _prune_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # Drain the log queue: send the sentinel and wait for the writer to finish.
        if _log_queue is not None:
            await _log_queue.put(None)
        if _log_writer_task is not None:
            try:
                await _log_writer_task
            except asyncio.CancelledError:
                pass
        _log_queue = None


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


def _evict_to_max_size(
    cache: collections.OrderedDict,
    max_items: int,
) -> None:
    """Drop the oldest entries until the cache is within max_items."""
    while len(cache) > max_items:
        cache.popitem(last=False)


def _prune_by_ttl(
    cache: collections.OrderedDict[str, tuple[float, Any] | float],
    now: float,
) -> None:
    """Remove entries whose timestamp is older than CACHE_TTL_S."""
    expired = [
        key for key, value in cache.items()
        if now - (value[0] if isinstance(value, tuple) else value) > CACHE_TTL_S
    ]
    for key in expired:
        cache.pop(key, None)


def _prune_ordered_cache(
    cache: collections.OrderedDict[str, tuple[float, Any] | float],
    max_items: int,
    now: float | None = None,
) -> None:
    """Combined size + TTL prune (kept for recovery-path and backward compat)."""
    _prune_by_ttl(cache, now or time.time())
    _evict_to_max_size(cache, max_items)


def _cache_completion(request_id: str, response: CompletionResponse) -> None:
    _completed_cache[request_id] = (time.time(), response)
    _completed_cache.move_to_end(request_id)
    _evict_to_max_size(_completed_cache, COMPLETED_CACHE_MAX)


def _get_completion(request_id: str) -> CompletionResponse | None:
    cached = _completed_cache.get(request_id)
    if cached is None:
        return None
    _completed_cache.move_to_end(request_id)
    return cached[1]


def _record_fingerprint(request_id: str, prompt: str) -> None:
    fp = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    _request_fingerprints[request_id] = fp
    _request_fingerprints.move_to_end(request_id)
    while len(_request_fingerprints) > COMPLETED_CACHE_MAX:
        _request_fingerprints.popitem(last=False)


def _fingerprint_matches(request_id: str, prompt: str) -> bool:
    existing = _request_fingerprints.get(request_id)
    if existing is None:
        return True
    return existing == hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _mark_cancelled(request_id: str) -> None:
    _cancelled[request_id] = time.time()
    _cancelled.move_to_end(request_id)
    _evict_to_max_size(_cancelled, CANCELLED_CACHE_MAX)


def _is_cancelled(request_id: str) -> bool:
    return request_id in _cancelled


def _mark_active(request_id: str, worker_url: str) -> None:
    _active[request_id] = (time.time(), worker_url)
    _active.move_to_end(request_id)
    _evict_to_max_size(_active, ACTIVE_CACHE_MAX)


def _get_active_worker(request_id: str) -> str | None:
    active = _active.get(request_id)
    if active is None:
        return None
    _active.move_to_end(request_id)
    return active[1]


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _write_log_entry(event: dict[str, Any]) -> None:
    """Synchronous disk write — runs in a thread via the log writer loop."""
    _ensure_log_dir()
    if REQUEST_LOG_MAX_BYTES > 0 and os.path.exists(REQUEST_LOG) and os.path.getsize(REQUEST_LOG) > REQUEST_LOG_MAX_BYTES:
        rotated = f"{REQUEST_LOG}.1"
        if os.path.exists(rotated):
            os.remove(rotated)
        os.replace(REQUEST_LOG, rotated)
    with open(REQUEST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _append_log(event: dict[str, Any]) -> None:
    """Enqueue a log entry for the background writer.

    Falls back to a synchronous write when called outside of a running lifespan
    (e.g. unit tests that do not start the server).
    """
    if _log_queue is not None:
        _log_queue.put_nowait(event)
    else:
        _write_log_entry(event)


async def _log_writer_loop() -> None:
    """Fix 6: drain _log_queue and write entries to disk via asyncio.to_thread.

    Sending None is the sentinel that signals clean shutdown.
    """
    while True:
        event = await _log_queue.get()  # type: ignore[union-attr]
        if event is None:
            break
        await asyncio.to_thread(_write_log_entry, event)


async def _prune_loop() -> None:
    """Fix 8: periodically evict TTL-expired entries from all caches.

    Keeps the TTL sweep off the request hot-path; size eviction still happens
    inline on every write so caches never grow unboundedly between sweeps.
    """
    while True:
        await asyncio.sleep(CACHE_PRUNE_INTERVAL_S)
        now = time.time()
        _prune_by_ttl(_completed_cache, now)
        _prune_by_ttl(_cancelled, now)
        _prune_by_ttl(_active, now)


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
    """Fix 7: circuit-breaker health loop with failure/recovery hysteresis.

    A healthy worker is marked unhealthy only after WORKER_FAILURE_THRESHOLD
    consecutive failed checks.  An unhealthy worker is only re-admitted after
    WORKER_RECOVERY_THRESHOLD consecutive successful checks, preventing a
    flapping worker from immediately rejoining the pool.
    """
    while True:
        async with httpx.AsyncClient(timeout=2.0) as client:
            for w in _workers:
                try:
                    r = await client.get(f"{w.url}/healthz")
                    if r.status_code == 200:
                        w.consecutive_failures = 0
                        w.consecutive_successes += 1
                        w.last_ok_unix_ms = int(time.time() * 1000)
                        if not w.healthy and w.consecutive_successes >= WORKER_RECOVERY_THRESHOLD:
                            w.healthy = True
                            _log.info(
                                "worker recovered",
                                extra={"worker_url": w.url, "after_successes": w.consecutive_successes},
                            )
                    else:
                        w.consecutive_successes = 0
                        w.consecutive_failures += 1
                        if w.healthy and w.consecutive_failures >= WORKER_FAILURE_THRESHOLD:
                            w.healthy = False
                            _log.warning(
                                "worker marked unhealthy",
                                extra={"worker_url": w.url, "consecutive_failures": w.consecutive_failures},
                            )
                except Exception:
                    w.consecutive_successes = 0
                    w.consecutive_failures += 1
                    if w.healthy and w.consecutive_failures >= WORKER_FAILURE_THRESHOLD:
                        w.healthy = False
                        _log.warning(
                            "worker marked unhealthy after health check exception",
                            extra={"worker_url": w.url, "consecutive_failures": w.consecutive_failures},
                            exc_info=True,
                        )
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
) -> AsyncGenerator[str, None]:
    generated: list[str] = []
    first_token_ttft_ms: float | None = None
    stream_start = time.perf_counter()
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
                        _record_fingerprint(request_id, prompt)
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
        _log.warning("worker stream timed out", extra={"request_id": request_id, "worker_url": worker.url})
        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "timed_out", "request_id": request_id})
    except Exception:
        _log.error("worker stream failed unexpectedly", extra={"request_id": request_id, "worker_url": worker.url}, exc_info=True)
        _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "failed", "request_id": request_id})
        raise
    finally:
        _active.pop(request_id, None)
        worker.inflight = max(0, worker.inflight - 1)


async def _worker_stream_with_retries(req: ChatRequest, prompt: str, request_id: str, deadline_ms: int | None) -> AsyncGenerator[str, None]:
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
            worker.consecutive_successes = 0
            skipped_workers.add(worker.url)
            _metrics["retry_attempts_total"] += 1
            _log.warning(
                "worker stream transport failed",
                extra={"request_id": request_id, "worker_url": worker.url, "emitted_tokens": emitted_tokens, "error": str(exc)},
            )
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
async def chat_completions(req: ChatRequest) -> CompletionResponse | StreamingResponse:
    now_ms = int(time.time() * 1000)
    deadline_ms = req.deadline_ms
    if deadline_ms is not None and deadline_ms <= now_ms:
        raise HTTPException(status_code=408, detail="Deadline already expired")

    prompt = "\n".join(m.content for m in req.messages)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long: {len(prompt)} chars exceeds limit of {MAX_PROMPT_CHARS}",
        )

    request_id = req.request_id or uuid.uuid4().hex
    _metrics["requests_total"] += 1

    # Hold _admission_lock while checking replay-safety conditions so two concurrent
    # requests sharing the same request_id cannot both miss the cache and double-dispatch.
    async with _admission_lock:
        cached = _get_completion(request_id)
        if cached is not None:
            if not _fingerprint_matches(request_id, prompt):
                _log.warning("request_id reused with different prompt content", extra={"request_id": request_id})
            return cached
        if _is_cancelled(request_id):
            raise HTTPException(status_code=499, detail="Request previously cancelled")

    if req.stream:
        _metrics["stream_requests_total"] += 1
        return StreamingResponse(
            _worker_stream_with_retries(req, prompt, request_id, deadline_ms),
            media_type="text/event-stream",
        )
    _metrics["nonstream_requests_total"] += 1

    skipped_workers: set[str] = set()
    worker: WorkerState | None = None
    while True:
        # Atomically select a worker and mark the request active so inflight
        # counts and the active map are never observed in an inconsistent state.
        async with _admission_lock:
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
            worker.consecutive_successes = 0
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
            _log.error("non-stream request failed unexpectedly", extra={"request_id": request_id, "worker_url": worker.url}, exc_info=True)
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
    _record_fingerprint(request_id, prompt)
    _append_log({"ts_unix_ms": int(time.time() * 1000), "event": "completed", "request_id": request_id, "response": result})
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
                _log.warning("failed to forward cancel to worker", extra={"request_id": request_id, "worker_url": worker_url}, exc_info=True)
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
            except Exception as exc:
                _log.warning("failed to fetch worker metrics", extra={"worker_url": w.url}, exc_info=True)
                entry["error"] = str(exc)
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
        "current_inflight_requests": len(_active),
        "workers_inflight_total": sum(w.inflight for w in _workers),
        "ttft_ms_p50": _pctl(ttft_values, 0.50),
        "ttft_ms_p95": _pctl(ttft_values, 0.95),
        "ttft_sample_count": len(ttft_values),
    }
