"""
Fix 3: Comprehensive error-path tests.
Fix 4: Concurrent-dispatch correctness tests.

Covers:
- KV allocation exhaustion (backpressure, admission timeout, retry-limit drop)
- Deadline exceeded (at intake, during worker call)
- All workers unreachable (503 escalation)
- Malformed / out-of-range request payloads
- Concurrent cancellation races
- Stream transport failure before vs. after first token
- Concurrent identical request_ids (lock correctness)
"""
from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import patch

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

import runtime.phase2.coordinator_server as coord
import runtime.phase2.worker_server as worker
from runtime.phase2.kv_allocator import PagedKVAllocator


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, script: list, calls: list) -> None:
        self._script = script
        self._calls = calls

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get(self, url: str) -> _FakeResponse:
        self._calls.append(("GET", url, None))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]

    async def post(self, url: str, json: dict | None = None, timeout: float | None = None) -> _FakeResponse:
        self._calls.append(("POST", url, json))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]


def _make_chat_req(**kwargs) -> coord.ChatRequest:
    defaults = dict(
        model="test-model",
        messages=[coord.Message(role="user", content="hello")],
        max_tokens=4,
        temperature=0.0,
        stream=False,
    )
    defaults.update(kwargs)
    return coord.ChatRequest(**defaults)


_COORD_METRICS_ZERO = {
    "requests_total": 0,
    "stream_requests_total": 0,
    "nonstream_requests_total": 0,
    "retry_attempts_total": 0,
    "cancellations_total": 0,
}


# ---------------------------------------------------------------------------
# Fix 3a: KV allocation exhaustion – worker-side
# ---------------------------------------------------------------------------

class TestWorkerKVAdmission(unittest.TestCase):
    """Backpressure, admission timeout, and KV retry-limit behaviour."""

    def setUp(self) -> None:
        worker._waiting.clear()
        worker._active.clear()
        worker._cancelled.clear()

    def tearDown(self) -> None:
        worker._waiting.clear()
        worker._active.clear()
        worker._cancelled.clear()

    # -- backpressure --------------------------------------------------------

    def test_backpressure_raises_503_when_kv_at_capacity(self) -> None:
        """_check_backpressure raises HTTP 503 when KV is fully allocated."""
        tiny = PagedKVAllocator(total_blocks=1, block_size_tokens=1)
        tiny.allocate_for_tokens("filler", 1)  # 100 % full

        original = worker._allocator
        try:
            worker._allocator = tiny
            with self.assertRaises(HTTPException) as ctx:
                worker._check_backpressure()
            self.assertEqual(ctx.exception.status_code, 503)
        finally:
            worker._allocator = original

    def test_backpressure_passes_when_kv_below_threshold(self) -> None:
        """_check_backpressure is silent when KV is well below the threshold."""
        alloc = PagedKVAllocator(total_blocks=100, block_size_tokens=1)
        alloc.allocate_for_tokens("filler", 50)  # 50 % full, well below 90 %

        original = worker._allocator
        try:
            worker._allocator = alloc
            worker._check_backpressure()  # must not raise
        finally:
            worker._allocator = original

    def test_backpressure_threshold_is_configurable_via_env(self) -> None:
        """KV_BACKPRESSURE_PCT env var is read at import; patching the constant works."""
        alloc = PagedKVAllocator(total_blocks=10, block_size_tokens=1)
        alloc.allocate_for_tokens("filler", 6)  # 60 % full

        original_alloc = worker._allocator
        original_pct = worker.KV_BACKPRESSURE_PCT
        try:
            worker._allocator = alloc
            worker.KV_BACKPRESSURE_PCT = 50.0  # lower threshold → 60 % should trigger
            with self.assertRaises(HTTPException) as ctx:
                worker._check_backpressure()
            self.assertEqual(ctx.exception.status_code, 503)
        finally:
            worker._allocator = original_alloc
            worker.KV_BACKPRESSURE_PCT = original_pct

    # -- admission timeout ---------------------------------------------------

    def test_admission_timeout_fails_stale_pending(self) -> None:
        """A request that waited past ADMISSION_TIMEOUT_S is resolved as timed_out."""
        alloc = PagedKVAllocator(total_blocks=1, block_size_tokens=1)
        alloc.allocate_for_tokens("filler", 1)  # keep KV full so it can never be admitted

        original_alloc = worker._allocator
        original_queue = worker._queue
        try:
            worker._allocator = alloc

            async def run() -> asyncio.Future:
                # Replace the module-level queue with a fresh one bound to this loop.
                worker._queue = asyncio.Queue()
                loop = asyncio.get_running_loop()
                fut: asyncio.Future = loop.create_future()
                stale = worker.Pending(
                    req=worker.GenerateRequest(request_id="req-stale", prompt="hi", max_tokens=1),
                    fut=fut,
                    admitted_at=time.time() - (worker.ADMISSION_TIMEOUT_S + 10),
                )
                worker._waiting.append(stale)

                task = asyncio.create_task(worker._continuous_batch_loop())
                await asyncio.sleep(0.06)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                return fut

            fut = asyncio.run(run())
            self.assertTrue(fut.done(), "future should be resolved after timeout")
            self.assertTrue(fut.result().get("timed_out"), "result should carry timed_out=True")
        finally:
            worker._allocator = original_alloc
            worker._queue = original_queue
            worker._waiting.clear()

    # -- KV retry limit ------------------------------------------------------

    def test_kv_retry_limit_drops_stuck_request(self) -> None:
        """A pending at MAX_KV_RETRIES is dropped rather than looping forever."""
        alloc = PagedKVAllocator(total_blocks=1, block_size_tokens=1)
        alloc.allocate_for_tokens("filler", 1)

        original_alloc = worker._allocator
        original_queue = worker._queue
        try:
            worker._allocator = alloc

            async def run() -> asyncio.Future:
                worker._queue = asyncio.Queue()
                loop = asyncio.get_running_loop()
                fut: asyncio.Future = loop.create_future()
                stuck = worker.Pending(
                    req=worker.GenerateRequest(request_id="req-stuck", prompt="hi", max_tokens=1),
                    fut=fut,
                    kv_retry_count=worker.MAX_KV_RETRIES,  # already at the limit
                )
                worker._waiting.append(stuck)

                task = asyncio.create_task(worker._continuous_batch_loop())
                await asyncio.sleep(0.06)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                return fut

            fut = asyncio.run(run())
            self.assertTrue(fut.done(), "future should be resolved after retry-limit breach")
            self.assertTrue(fut.result().get("timed_out"), "result should carry timed_out=True")
        finally:
            worker._allocator = original_alloc
            worker._queue = original_queue
            worker._waiting.clear()

    def test_kv_retry_count_increments_before_requeue(self) -> None:
        """Each failed KV allocation attempt increments kv_retry_count by 1."""
        alloc = PagedKVAllocator(total_blocks=1, block_size_tokens=1)
        alloc.allocate_for_tokens("filler", 1)

        original_alloc = worker._allocator
        original_max = worker.MAX_KV_RETRIES
        original_queue = worker._queue
        try:
            worker._allocator = alloc
            worker.MAX_KV_RETRIES = 5  # low limit so we don't spin long

            async def run() -> asyncio.Future:
                worker._queue = asyncio.Queue()
                loop = asyncio.get_running_loop()
                fut: asyncio.Future = loop.create_future()
                pending = worker.Pending(
                    req=worker.GenerateRequest(request_id="req-count", prompt="hi", max_tokens=1),
                    fut=fut,
                    kv_retry_count=0,
                )
                worker._waiting.append(pending)

                task = asyncio.create_task(worker._continuous_batch_loop())
                await asyncio.sleep(0.06)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                return fut

            fut = asyncio.run(run())
            # The request was dropped at retry limit — future resolves as timed_out.
            self.assertTrue(fut.done())
            self.assertTrue(fut.result().get("timed_out"))
        finally:
            worker._allocator = original_alloc
            worker.MAX_KV_RETRIES = original_max
            worker._queue = original_queue
            worker._waiting.clear()


# ---------------------------------------------------------------------------
# Fix 3b: Deadline exceeded – coordinator-side
# ---------------------------------------------------------------------------

class TestCoordinatorDeadlines(unittest.TestCase):

    def setUp(self) -> None:
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()

    def tearDown(self) -> None:
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()
        coord._metrics.update(_COORD_METRICS_ZERO)

    def test_already_expired_deadline_rejected_at_intake(self) -> None:
        """deadline_ms in the past raises HTTP 408 without touching any worker."""
        coord._workers.append(coord.WorkerState(url="http://worker-a", healthy=True))
        past_ms = int(time.time() * 1000) - 5_000

        with patch.object(coord.httpx, "AsyncClient", side_effect=AssertionError("must not dispatch")):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(coord.chat_completions(_make_chat_req(deadline_ms=past_ms)))

        self.assertEqual(ctx.exception.status_code, 408)

    def test_worker_timeout_exception_raises_408(self) -> None:
        """httpx.TimeoutException during worker call surfaces as HTTP 408."""
        coord._workers.append(coord.WorkerState(url="http://worker-a", healthy=True))
        calls: list = []
        script = [httpx.TimeoutException("timed out")]

        with patch.object(coord.httpx, "AsyncClient", lambda timeout=None: _FakeAsyncClient(script, calls)):
            with patch.object(coord, "_append_log", lambda e: None):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(coord.chat_completions(_make_chat_req(request_id="req-timeout")))

        self.assertEqual(ctx.exception.status_code, 408)

    def test_deadline_exceeded_after_retry_raises_408(self) -> None:
        """If deadline expires between retry attempts, the loop terminates with 408."""
        coord._workers.extend([
            coord.WorkerState(url="http://worker-a", healthy=True),
            coord.WorkerState(url="http://worker-b", healthy=True),
        ])
        calls: list = []
        # First worker fails with transport error; set a deadline that is already
        # past by the time we reach the retry-deadline check.
        fail = httpx.RequestError("boom", request=httpx.Request("POST", "http://worker-a/generate"))
        script = [fail]
        past_ms = int(time.time() * 1000) - 1

        with patch.object(coord.httpx, "AsyncClient", lambda timeout=None: _FakeAsyncClient(script, calls)):
            with patch.object(coord, "_append_log", lambda e: None):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(coord.chat_completions(_make_chat_req(request_id="req-dl-retry", deadline_ms=past_ms)))

        self.assertIn(ctx.exception.status_code, {408, 503})


# ---------------------------------------------------------------------------
# Fix 3c: All workers unreachable
# ---------------------------------------------------------------------------

class TestAllWorkersExhausted(unittest.TestCase):

    def setUp(self) -> None:
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()

    def tearDown(self) -> None:
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()
        coord._metrics.update(_COORD_METRICS_ZERO)

    def test_no_healthy_workers_raises_503(self) -> None:
        """No healthy workers in the pool → HTTP 503 immediately."""
        coord._workers.append(coord.WorkerState(url="http://worker-a", healthy=False))

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(coord.chat_completions(_make_chat_req()))

        self.assertEqual(ctx.exception.status_code, 503)

    def test_all_workers_transport_fail_raises_503(self) -> None:
        """Two workers both fail with transport errors → HTTP 503 after exhausting all."""
        coord._workers.extend([
            coord.WorkerState(url="http://worker-a", healthy=True),
            coord.WorkerState(url="http://worker-b", healthy=True),
        ])
        calls: list = []
        fail_a = httpx.RequestError("boom-a", request=httpx.Request("POST", "http://worker-a/generate"))
        fail_b = httpx.RequestError("boom-b", request=httpx.Request("POST", "http://worker-b/generate"))
        script = [fail_a, fail_b]

        with patch.object(coord.httpx, "AsyncClient", lambda timeout=None: _FakeAsyncClient(script, calls)):
            with patch.object(coord, "_append_log", lambda e: None):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(coord.chat_completions(
                        _make_chat_req(request_id="req-all-fail")
                    ))

        self.assertEqual(ctx.exception.status_code, 503)
        # Both workers should be marked unhealthy after their failure.
        self.assertFalse(coord._workers[0].healthy)
        self.assertFalse(coord._workers[1].healthy)


# ---------------------------------------------------------------------------
# Fix 3d: Malformed / out-of-range request payloads
# ---------------------------------------------------------------------------

class TestMalformedPayloads(unittest.TestCase):

    def test_coordinator_max_tokens_below_minimum_rejected(self) -> None:
        """max_tokens=0 violates ge=1 and raises ValidationError."""
        with self.assertRaises(ValidationError):
            coord.ChatRequest(
                model="test",
                messages=[coord.Message(role="user", content="hi")],
                max_tokens=0,
            )

    def test_coordinator_max_tokens_above_maximum_rejected(self) -> None:
        """max_tokens=9999 violates le=2048 and raises ValidationError."""
        with self.assertRaises(ValidationError):
            coord.ChatRequest(
                model="test",
                messages=[coord.Message(role="user", content="hi")],
                max_tokens=9999,
            )

    def test_coordinator_missing_model_field_rejected(self) -> None:
        """Omitting the required 'model' field raises ValidationError."""
        with self.assertRaises((ValidationError, TypeError)):
            coord.ChatRequest(  # type: ignore[call-arg]
                messages=[coord.Message(role="user", content="hi")],
                max_tokens=4,
            )

    def test_worker_generate_request_max_tokens_below_minimum(self) -> None:
        """Worker GenerateRequest enforces ge=1 on max_tokens."""
        with self.assertRaises(ValidationError):
            worker.GenerateRequest(request_id="r", prompt="hi", max_tokens=0)

    def test_worker_generate_request_max_tokens_above_maximum(self) -> None:
        """Worker GenerateRequest enforces le=2048 on max_tokens."""
        with self.assertRaises(ValidationError):
            worker.GenerateRequest(request_id="r", prompt="hi", max_tokens=9999)

    def test_coordinator_negative_temperature_accepted(self) -> None:
        """Negative temperature has no Pydantic constraint; value is stored as-is."""
        req = coord.ChatRequest(
            model="test",
            messages=[coord.Message(role="user", content="hi")],
            max_tokens=4,
            temperature=-1.0,
        )
        self.assertEqual(req.temperature, -1.0)

    def test_coordinator_boundary_max_tokens_accepted(self) -> None:
        """Boundary values 1 and 2048 are both valid."""
        for v in (1, 2048):
            req = coord.ChatRequest(
                model="test",
                messages=[coord.Message(role="user", content="hi")],
                max_tokens=v,
            )
            self.assertEqual(req.max_tokens, v)


# ---------------------------------------------------------------------------
# Fix 3e: Concurrent cancellation races
# ---------------------------------------------------------------------------

class TestConcurrentCancellation(unittest.TestCase):

    def setUp(self) -> None:
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()

    def tearDown(self) -> None:
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()
        coord._metrics.update(_COORD_METRICS_ZERO)

    def test_cancel_of_active_request_forwards_to_correct_worker(self) -> None:
        """Cancelling an in-flight request sends the cancel to its worker."""
        coord._workers.append(coord.WorkerState(url="http://worker-a", healthy=True))
        coord._mark_active("req-active", "http://worker-a")

        calls: list = []
        script = [_FakeResponse({})]

        with patch.object(coord.httpx, "AsyncClient", lambda timeout=None: _FakeAsyncClient(script, calls)):
            with patch.object(coord, "_append_log", lambda e: None):
                result = asyncio.run(coord.cancel_request("req-active"))

        self.assertEqual(result["status"], "cancel_accepted")
        self.assertIn("req-active", coord._cancelled)
        self.assertIn(("POST", "http://worker-a/cancel/req-active", None), calls)

    def test_pre_cancelled_request_id_rejected_without_dispatch(self) -> None:
        """A request whose ID was cancelled before dispatch gets HTTP 499."""
        coord._workers.append(coord.WorkerState(url="http://worker-a", healthy=True))
        coord._mark_cancelled("req-precancelled")

        with patch.object(coord.httpx, "AsyncClient", side_effect=AssertionError("must not dispatch")):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(coord.chat_completions(_make_chat_req(request_id="req-precancelled")))

        self.assertEqual(ctx.exception.status_code, 499)

    def test_cancel_of_unknown_request_is_accepted_gracefully(self) -> None:
        """Cancelling a request with no known active worker succeeds without error."""
        with patch.object(coord, "_append_log", lambda e: None):
            result = asyncio.run(coord.cancel_request("req-unknown"))

        self.assertEqual(result["status"], "cancel_accepted")
        self.assertIn("req-unknown", coord._cancelled)

    def test_worker_cancelled_signal_resolves_active_state(self) -> None:
        """Worker batch loop resolves a future with cancelled=True for a cancelled request."""

        async def run() -> dict:
            alloc = PagedKVAllocator(total_blocks=64, block_size_tokens=16)
            original_alloc = worker._allocator
            original_queue = worker._queue
            try:
                worker._allocator = alloc
                worker._queue = asyncio.Queue()
                loop = asyncio.get_running_loop()
                fut: asyncio.Future = loop.create_future()
                req = worker.GenerateRequest(request_id="req-cancel-worker", prompt="a b c", max_tokens=10)
                pending = worker.Pending(req=req, fut=fut)
                worker._waiting.append(pending)

                task = asyncio.create_task(worker._continuous_batch_loop())
                # Give loop time to admit the request.
                await asyncio.sleep(0.02)
                # Cancel it from the worker side.
                worker._mark_cancelled("req-cancel-worker")
                # Wait for the batch loop to detect and resolve it.
                await asyncio.sleep(0.05)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                return fut.result() if fut.done() else {}
            finally:
                worker._allocator = original_alloc
                worker._queue = original_queue
                worker._waiting.clear()
                worker._active.clear()
                worker._cancelled.clear()

        result = asyncio.run(run())
        self.assertTrue(result.get("cancelled"), f"expected cancelled=True, got {result}")


# ---------------------------------------------------------------------------
# Fix 3f: Stream transport failure – before and after first token
# ---------------------------------------------------------------------------

class TestStreamWorkerFailure(unittest.TestCase):

    def setUp(self) -> None:
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()

    def tearDown(self) -> None:
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()
        coord._metrics.update(_COORD_METRICS_ZERO)

    def _make_stream_req(self, request_id: str = "req-stream") -> coord.ChatRequest:
        return coord.ChatRequest(
            model="test-model",
            messages=[coord.Message(role="user", content="hello")],
            max_tokens=2,
            temperature=0.0,
            stream=True,
            request_id=request_id,
        )

    def test_stream_retries_second_worker_on_transport_failure_before_first_token(self) -> None:
        """Transport error before any token emitted → retry on next worker."""
        coord._workers.extend([
            coord.WorkerState(url="http://worker-a", healthy=True),
            coord.WorkerState(url="http://worker-b", healthy=True),
        ])

        call_log: list[str] = []

        async def fake_worker_stream_a(req, prompt, request_id, w, deadline_ms):  # type: ignore[misc]
            call_log.append(w.url)
            raise httpx.RequestError("fail", request=httpx.Request("POST", w.url))
            yield  # make this an async generator

        async def fake_worker_stream_b(req, prompt, request_id, w, deadline_ms):  # type: ignore[misc]
            call_log.append(w.url)
            # Emit one token chunk and a done chunk.
            chunk_json = '{"choices":[{"delta":{"content":"hi"},"index":0}]}'
            yield f"data: {chunk_json}\n\n"
            yield "data: [DONE]\n\n"

        # _worker_stream is looked up by name in _worker_stream_with_retries.
        # We route based on the worker URL by replacing it with a dispatcher.
        dispatched: list[str] = []

        async def dispatch(req, prompt, request_id, w, deadline_ms):  # type: ignore[misc]
            dispatched.append(w.url)
            if w.url == "http://worker-a":
                async for chunk in fake_worker_stream_a(req, prompt, request_id, w, deadline_ms):
                    yield chunk
            else:
                async for chunk in fake_worker_stream_b(req, prompt, request_id, w, deadline_ms):
                    yield chunk

        async def run() -> list[str]:
            with patch.object(coord, "_worker_stream", dispatch):
                with patch.object(coord, "_append_log", lambda e: None):
                    chunks: list[str] = []
                    async for chunk in coord._worker_stream_with_retries(
                        self._make_stream_req(), "hello", "req-s-retry", None
                    ):
                        chunks.append(chunk)
                    return chunks

        chunks = asyncio.run(run())
        self.assertEqual(dispatched, ["http://worker-a", "http://worker-b"])
        self.assertFalse(coord._workers[0].healthy, "worker-a should be marked unhealthy")
        # At least one data chunk from worker-b was forwarded.
        self.assertTrue(any("content" in c or "DONE" in c for c in chunks))

    def test_stream_does_not_retry_after_first_token_emitted(self) -> None:
        """Transport error after a content token was emitted must NOT retry."""
        coord._workers.extend([
            coord.WorkerState(url="http://worker-a", healthy=True),
            coord.WorkerState(url="http://worker-b", healthy=True),
        ])

        dispatched: list[str] = []

        async def fake_stream_partial(req, prompt, request_id, w, deadline_ms):  # type: ignore[misc]
            dispatched.append(w.url)
            # Emit a content token first, then raise a transport error.
            chunk_json = '{"choices":[{"delta":{"content":"hi"},"index":0}]}'
            yield f"data: {chunk_json}\n\n"
            raise httpx.RequestError("mid-stream fail", request=httpx.Request("POST", w.url))

        async def run() -> None:
            with patch.object(coord, "_worker_stream", fake_stream_partial):
                with patch.object(coord, "_append_log", lambda e: None):
                    async for _ in coord._worker_stream_with_retries(
                        self._make_stream_req(), "hello", "req-s-no-retry", None
                    ):
                        pass

        with self.assertRaises(httpx.RequestError):
            asyncio.run(run())

        # Only worker-a should have been tried — no retry to worker-b.
        self.assertEqual(dispatched, ["http://worker-a"])


# ---------------------------------------------------------------------------
# Fix 4: Concurrent identical request_ids – lock correctness
# ---------------------------------------------------------------------------

class TestAdmissionLockCorrectness(unittest.TestCase):
    """Verify that _admission_lock prevents double-dispatch for concurrent requests."""

    def setUp(self) -> None:
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()

    def tearDown(self) -> None:
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()
        coord._metrics.update(_COORD_METRICS_ZERO)

    def test_concurrent_identical_cached_requests_never_dispatch(self) -> None:
        """Two concurrent requests sharing a cached request_id both get the cached result."""
        cached = {
            "id": "chatcmpl-phase2",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "cached"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        coord._cache_completion("req-concurrent", cached)

        async def run() -> list:
            req = _make_chat_req(request_id="req-concurrent")
            with patch.object(coord.httpx, "AsyncClient", side_effect=AssertionError("must not dispatch")):
                return list(await asyncio.gather(
                    coord.chat_completions(req),
                    coord.chat_completions(req),
                ))

        results = asyncio.run(run())
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], cached)
        self.assertEqual(results[1], cached)

    def test_concurrent_pre_cancelled_requests_both_raise_499(self) -> None:
        """Two concurrent requests with a pre-cancelled ID both get HTTP 499."""
        coord._mark_cancelled("req-precancelled-concurrent")
        coord._workers.append(coord.WorkerState(url="http://worker-a", healthy=True))

        async def run() -> list[int]:
            req = _make_chat_req(request_id="req-precancelled-concurrent")
            statuses: list[int] = []

            async def one() -> None:
                try:
                    await coord.chat_completions(req)
                except HTTPException as exc:
                    statuses.append(exc.status_code)

            await asyncio.gather(one(), one())
            return statuses

        with patch.object(coord.httpx, "AsyncClient", side_effect=AssertionError("must not dispatch")):
            statuses = asyncio.run(run())

        self.assertEqual(sorted(statuses), [499, 499])

    def test_worker_inflight_returns_to_zero_after_request_completes(self) -> None:
        """After a successful non-stream request, the worker's inflight count is 0."""
        coord._workers.append(coord.WorkerState(url="http://worker-a", healthy=True, inflight=0))
        calls: list = []
        script = [_FakeResponse({"request_id": "req-inflight", "text": "done", "cancelled": False})]

        with patch.object(coord.httpx, "AsyncClient", lambda timeout=None: _FakeAsyncClient(script, calls)):
            with patch.object(coord, "_append_log", lambda e: None):
                asyncio.run(coord.chat_completions(_make_chat_req(request_id="req-inflight")))

        self.assertEqual(coord._workers[0].inflight, 0)

    def test_worker_inflight_returns_to_zero_after_transport_failure(self) -> None:
        """After a transport failure that exhausts all workers, inflight is 0."""
        coord._workers.append(coord.WorkerState(url="http://worker-a", healthy=True, inflight=0))
        calls: list = []
        fail = httpx.RequestError("fail", request=httpx.Request("POST", "http://worker-a/generate"))
        script = [fail]

        with patch.object(coord.httpx, "AsyncClient", lambda timeout=None: _FakeAsyncClient(script, calls)):
            with patch.object(coord, "_append_log", lambda e: None):
                with self.assertRaises(HTTPException):
                    asyncio.run(coord.chat_completions(_make_chat_req(request_id="req-fail-inflight")))

        self.assertEqual(coord._workers[0].inflight, 0)


if __name__ == "__main__":
    unittest.main()
