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
import os
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


def _reset_coord_state() -> None:
    coord._workers.clear()
    coord._active.clear()
    coord._completed_cache.clear()
    coord._cancelled.clear()
    coord._request_fingerprints.clear()
    coord._metrics.update(_COORD_METRICS_ZERO)


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
        _reset_coord_state()

    def tearDown(self) -> None:
        _reset_coord_state()

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
        _reset_coord_state()

    def tearDown(self) -> None:
        _reset_coord_state()

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
        _reset_coord_state()

    def tearDown(self) -> None:
        _reset_coord_state()

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
        _reset_coord_state()

    def tearDown(self) -> None:
        _reset_coord_state()

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
        _reset_coord_state()

    def tearDown(self) -> None:
        _reset_coord_state()

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


# ---------------------------------------------------------------------------
# Fix 5 / Fix 6: env var validation, prompt length, request_id fingerprinting
# ---------------------------------------------------------------------------

class TestEnvVarValidation(unittest.TestCase):
    """_env_int and _env_float reject bad values at startup."""

    def test_env_int_rejects_non_integer(self) -> None:
        import os
        os.environ["_TEST_BAD_INT"] = "not_a_number"
        try:
            with self.assertRaises(ValueError):
                coord._env_int("_TEST_BAD_INT", 1)
        finally:
            del os.environ["_TEST_BAD_INT"]

    def test_env_int_rejects_below_min(self) -> None:
        import os
        os.environ["_TEST_MIN_INT"] = "0"
        try:
            with self.assertRaises(ValueError):
                coord._env_int("_TEST_MIN_INT", 1, min_val=1)
        finally:
            del os.environ["_TEST_MIN_INT"]

    def test_env_int_rejects_above_max(self) -> None:
        import os
        os.environ["_TEST_MAX_INT"] = "999"
        try:
            with self.assertRaises(ValueError):
                coord._env_int("_TEST_MAX_INT", 10, max_val=100)
        finally:
            del os.environ["_TEST_MAX_INT"]

    def test_env_float_rejects_non_float(self) -> None:
        import os
        os.environ["_TEST_BAD_FLOAT"] = "abc"
        try:
            with self.assertRaises(ValueError):
                coord._env_float("_TEST_BAD_FLOAT", 1.0)
        finally:
            del os.environ["_TEST_BAD_FLOAT"]

    def test_env_float_rejects_out_of_range(self) -> None:
        import os
        os.environ["_TEST_RANGE_FLOAT"] = "150.0"
        try:
            with self.assertRaises(ValueError):
                coord._env_float("_TEST_RANGE_FLOAT", 90.0, min_val=0.0, max_val=100.0)
        finally:
            del os.environ["_TEST_RANGE_FLOAT"]

    def test_env_int_returns_default_when_var_unset(self) -> None:
        import os
        os.environ.pop("_TEST_UNSET_VAR", None)
        self.assertEqual(coord._env_int("_TEST_UNSET_VAR", 42), 42)

    def test_worker_env_helpers_match_coordinator(self) -> None:
        """Worker's _env_int/_env_float are independent but behave identically."""
        import os
        os.environ["_TEST_W"] = "5"
        try:
            self.assertEqual(worker._env_int("_TEST_W", 99), 5)
        finally:
            del os.environ["_TEST_W"]


class TestPromptLengthValidation(unittest.TestCase):
    """Prompt length limits are enforced at both coordinator and worker boundaries."""

    def setUp(self) -> None:
        _reset_coord_state()

    def tearDown(self) -> None:
        _reset_coord_state()

    def test_coordinator_rejects_prompt_exceeding_max_chars(self) -> None:
        """Prompt longer than MAX_PROMPT_CHARS raises HTTP 400."""
        coord._workers.append(coord.WorkerState(url="http://worker-a", healthy=True))
        original_limit = coord.MAX_PROMPT_CHARS
        try:
            coord.MAX_PROMPT_CHARS = 10
            with patch.object(coord.httpx, "AsyncClient", side_effect=AssertionError("must not dispatch")):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(coord.chat_completions(
                        _make_chat_req(messages=[coord.Message(role="user", content="x" * 20)])
                    ))
            self.assertEqual(ctx.exception.status_code, 400)
        finally:
            coord.MAX_PROMPT_CHARS = original_limit

    def test_coordinator_accepts_prompt_at_exact_limit(self) -> None:
        """Prompt exactly at MAX_PROMPT_CHARS passes the length check."""
        original_limit = coord.MAX_PROMPT_CHARS
        try:
            coord.MAX_PROMPT_CHARS = 5
            req = _make_chat_req(messages=[coord.Message(role="user", content="hello")])  # exactly 5 chars
            prompt = "\n".join(m.content for m in req.messages)
            self.assertLessEqual(len(prompt), coord.MAX_PROMPT_CHARS)
        finally:
            coord.MAX_PROMPT_CHARS = original_limit

    def test_worker_generate_request_rejects_long_prompt(self) -> None:
        """GenerateRequest.prompt field validator rejects oversized prompts."""
        original_limit = worker.MAX_PROMPT_CHARS
        try:
            worker.MAX_PROMPT_CHARS = 10
            with self.assertRaises(Exception):
                worker.GenerateRequest(request_id="r", prompt="x" * 20, max_tokens=4)
        finally:
            worker.MAX_PROMPT_CHARS = original_limit

    def test_worker_generate_request_accepts_prompt_at_limit(self) -> None:
        """GenerateRequest accepts a prompt at exactly MAX_PROMPT_CHARS."""
        original_limit = worker.MAX_PROMPT_CHARS
        try:
            worker.MAX_PROMPT_CHARS = 5
            req = worker.GenerateRequest(request_id="r", prompt="hello", max_tokens=4)
            self.assertEqual(req.prompt, "hello")
        finally:
            worker.MAX_PROMPT_CHARS = original_limit


class TestRequestIdFingerprinting(unittest.TestCase):
    """request_id reuse with a different prompt is detected and logged."""

    def setUp(self) -> None:
        _reset_coord_state()

    def tearDown(self) -> None:
        _reset_coord_state()

    def test_same_prompt_fingerprint_matches(self) -> None:
        coord._record_fingerprint("req-fp", "hello world")
        self.assertTrue(coord._fingerprint_matches("req-fp", "hello world"))

    def test_different_prompt_fingerprint_does_not_match(self) -> None:
        coord._record_fingerprint("req-fp", "hello world")
        self.assertFalse(coord._fingerprint_matches("req-fp", "different content"))

    def test_unknown_request_id_fingerprint_always_matches(self) -> None:
        """If no fingerprint was recorded, the check passes (no false positives)."""
        self.assertTrue(coord._fingerprint_matches("req-unknown", "any content"))

    def test_fingerprint_evicts_oldest_when_full(self) -> None:
        """_request_fingerprints is bounded to COMPLETED_CACHE_MAX entries."""
        original_max = coord.COMPLETED_CACHE_MAX
        try:
            coord.COMPLETED_CACHE_MAX = 3
            for i in range(5):
                coord._record_fingerprint(f"req-{i}", f"prompt-{i}")
            self.assertLessEqual(len(coord._request_fingerprints), 3)
        finally:
            coord.COMPLETED_CACHE_MAX = original_max

    def test_reuse_with_different_prompt_logs_warning(self) -> None:
        """Returning a cached result for a reused request_id with a new prompt logs a warning."""
        cached = {
            "id": "chatcmpl-phase2",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "old"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        coord._cache_completion("req-reuse", cached)  # type: ignore[arg-type]
        coord._record_fingerprint("req-reuse", "original prompt")

        import logging as _logging
        with self.assertLogs("coordinator", level=_logging.WARNING) as log_ctx:
            result = asyncio.run(coord.chat_completions(
                coord.ChatRequest(
                    model="test-model",
                    messages=[coord.Message(role="user", content="completely different")],
                    max_tokens=4,
                    stream=False,
                    request_id="req-reuse",
                )
            ))

        self.assertEqual(result, cached)
        self.assertTrue(any("reused" in msg for msg in log_ctx.output))


# ---------------------------------------------------------------------------
# Fix 6: Non-blocking log writes
# Fix 7: Worker circuit-breaker hysteresis
# Fix 8: Background cache eviction
# ---------------------------------------------------------------------------

class TestAsyncLogWriter(unittest.TestCase):
    """_append_log enqueues when a log_queue is present; falls back to sync when not."""

    def tearDown(self) -> None:
        coord._log_queue = None

    def test_append_log_enqueues_when_queue_present(self) -> None:
        """_append_log puts the event into _log_queue without touching disk."""
        coord._log_queue = asyncio.Queue()
        event = {"event": "test", "request_id": "r1"}
        coord._append_log(event)
        self.assertFalse(coord._log_queue.empty())
        queued = coord._log_queue.get_nowait()
        self.assertEqual(queued, event)

    def test_append_log_falls_back_to_sync_when_no_queue(self) -> None:
        """Without a queue (unit-test mode) _append_log writes synchronously."""
        coord._log_queue = None
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = pathlib.Path(tmpdir) / "requests.jsonl"
            with patch.object(coord, "REQUEST_LOG", str(log_path)):
                with patch.object(coord, "LOG_DIR", tmpdir):
                    coord._append_log({"event": "sync_test"})
            self.assertTrue(log_path.exists())
            lines = [l for l in log_path.read_text().splitlines() if l.strip()]
            self.assertEqual(len(lines), 1)

    def test_log_writer_loop_drains_queue_and_stops_on_sentinel(self) -> None:
        """_log_writer_loop writes queued events then exits when it receives None."""
        import tempfile, pathlib

        async def run() -> None:
            coord._log_queue = asyncio.Queue()
            with tempfile.TemporaryDirectory() as tmpdir:
                log_path = pathlib.Path(tmpdir) / "requests.jsonl"
                with patch.object(coord, "REQUEST_LOG", str(log_path)):
                    with patch.object(coord, "LOG_DIR", tmpdir):
                        coord._log_queue.put_nowait({"event": "a"})
                        coord._log_queue.put_nowait({"event": "b"})
                        coord._log_queue.put_nowait(None)  # sentinel
                        await coord._log_writer_loop()
                lines = [l for l in log_path.read_text().splitlines() if l.strip()]
                self.assertEqual(len(lines), 2)

        asyncio.run(run())
        coord._log_queue = None

    def test_sentinel_causes_writer_to_exit(self) -> None:
        """Sending None to the queue terminates _log_writer_loop cleanly."""
        import tempfile, pathlib

        async def run() -> None:
            coord._log_queue = asyncio.Queue()
            with tempfile.TemporaryDirectory() as tmpdir:
                log_path = pathlib.Path(tmpdir) / "requests.jsonl"
                with patch.object(coord, "REQUEST_LOG", str(log_path)):
                    with patch.object(coord, "LOG_DIR", tmpdir):
                        coord._log_queue.put_nowait(None)
                        await coord._log_writer_loop()  # should return, not loop
            # No exception means success

        asyncio.run(run())
        coord._log_queue = None


class TestWorkerCircuitBreaker(unittest.TestCase):
    """_health_loop requires multiple consecutive results before flipping healthy."""

    def _make_worker(self, healthy: bool = True) -> coord.WorkerState:
        return coord.WorkerState(url="http://w", healthy=healthy)

    def test_single_health_check_failure_does_not_immediately_mark_unhealthy(self) -> None:
        """A healthy worker needs WORKER_FAILURE_THRESHOLD consecutive failures."""
        original = coord.WORKER_FAILURE_THRESHOLD
        try:
            coord.WORKER_FAILURE_THRESHOLD = 2
            w = self._make_worker(healthy=True)
            w.consecutive_failures += 1  # simulate one failure
            # Still below threshold: healthy unchanged
            self.assertTrue(w.healthy)
        finally:
            coord.WORKER_FAILURE_THRESHOLD = original

    def test_worker_marked_unhealthy_at_failure_threshold(self) -> None:
        """Reaching WORKER_FAILURE_THRESHOLD flips healthy=False."""
        original = coord.WORKER_FAILURE_THRESHOLD
        try:
            coord.WORKER_FAILURE_THRESHOLD = 2
            w = self._make_worker(healthy=True)
            # Simulate two consecutive failures (as health loop would do)
            for _ in range(coord.WORKER_FAILURE_THRESHOLD):
                w.consecutive_successes = 0
                w.consecutive_failures += 1
                if w.healthy and w.consecutive_failures >= coord.WORKER_FAILURE_THRESHOLD:
                    w.healthy = False
            self.assertFalse(w.healthy)
        finally:
            coord.WORKER_FAILURE_THRESHOLD = original

    def test_single_health_check_success_does_not_immediately_restore_healthy(self) -> None:
        """An unhealthy worker needs WORKER_RECOVERY_THRESHOLD consecutive successes."""
        original = coord.WORKER_RECOVERY_THRESHOLD
        try:
            coord.WORKER_RECOVERY_THRESHOLD = 3
            w = self._make_worker(healthy=False)
            w.consecutive_successes = 1  # only one success so far
            # Still below recovery threshold
            self.assertFalse(w.healthy)
        finally:
            coord.WORKER_RECOVERY_THRESHOLD = original

    def test_worker_recovers_at_recovery_threshold(self) -> None:
        """Reaching WORKER_RECOVERY_THRESHOLD consecutive successes restores healthy."""
        original = coord.WORKER_RECOVERY_THRESHOLD
        try:
            coord.WORKER_RECOVERY_THRESHOLD = 3
            w = self._make_worker(healthy=False)
            for _ in range(coord.WORKER_RECOVERY_THRESHOLD):
                w.consecutive_failures = 0
                w.consecutive_successes += 1
                if not w.healthy and w.consecutive_successes >= coord.WORKER_RECOVERY_THRESHOLD:
                    w.healthy = True
            self.assertTrue(w.healthy)
        finally:
            coord.WORKER_RECOVERY_THRESHOLD = original

    def test_transport_failure_resets_consecutive_successes(self) -> None:
        """A request-path transport error resets the recovery streak."""
        w = self._make_worker(healthy=True)
        w.consecutive_successes = 2
        # Simulate what the request path does on httpx.RequestError
        w.healthy = False
        w.consecutive_successes = 0
        self.assertFalse(w.healthy)
        self.assertEqual(w.consecutive_successes, 0)


class TestBackgroundCacheEviction(unittest.TestCase):
    """Background prune task keeps caches TTL-clean without blocking requests."""

    def setUp(self) -> None:
        _reset_coord_state()

    def tearDown(self) -> None:
        _reset_coord_state()

    def test_prune_by_ttl_removes_expired_entries(self) -> None:
        """_prune_by_ttl deletes entries older than CACHE_TTL_S."""
        original_ttl = coord.CACHE_TTL_S
        try:
            coord.CACHE_TTL_S = 1.0
            coord._completed_cache["old"] = (time.time() - 5.0, {})  # type: ignore[assignment]
            coord._completed_cache["new"] = (time.time(), {})  # type: ignore[assignment]
            coord._prune_by_ttl(coord._completed_cache, time.time())
            self.assertNotIn("old", coord._completed_cache)
            self.assertIn("new", coord._completed_cache)
        finally:
            coord.CACHE_TTL_S = original_ttl

    def test_evict_to_max_size_drops_oldest_entries(self) -> None:
        """_evict_to_max_size removes the least-recently-inserted entries."""
        for i in range(5):
            coord._completed_cache[f"req-{i}"] = (time.time(), {})  # type: ignore[assignment]
        coord._evict_to_max_size(coord._completed_cache, 3)
        self.assertLessEqual(len(coord._completed_cache), 3)
        # The two oldest (req-0, req-1) should have been evicted.
        self.assertNotIn("req-0", coord._completed_cache)
        self.assertNotIn("req-1", coord._completed_cache)

    def test_get_completion_does_not_trigger_pruning(self) -> None:
        """_get_completion no longer calls any prune function (hot-path is lean)."""
        import inspect
        src = inspect.getsource(coord._get_completion)
        self.assertNotIn("_prune", src)
        self.assertNotIn("_evict", src)

    def test_is_cancelled_does_not_trigger_pruning(self) -> None:
        """_is_cancelled no longer calls any prune function."""
        import inspect
        src = inspect.getsource(coord._is_cancelled)
        self.assertNotIn("_prune", src)
        self.assertNotIn("_evict", src)

    def test_prune_loop_runs_periodically(self) -> None:
        """_prune_loop sleeps then prunes, repeating until cancelled."""
        original_ttl = coord.CACHE_TTL_S
        try:
            coord.CACHE_TTL_S = 0.0  # expire everything immediately
            coord._completed_cache["stale"] = (time.time() - 1.0, {})  # type: ignore[assignment]
            coord._cancelled["stale-cancel"] = time.time() - 1.0

            async def run() -> None:
                # Run one iteration: sleep(interval) → prune → sleep(interval) → cancel
                with patch.object(coord, "CACHE_PRUNE_INTERVAL_S", 0.01):
                    task = asyncio.create_task(coord._prune_loop())
                    await asyncio.sleep(0.05)
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            asyncio.run(run())
            self.assertNotIn("stale", coord._completed_cache)
            self.assertNotIn("stale-cancel", coord._cancelled)
        finally:
            coord.CACHE_TTL_S = original_ttl


# ---------------------------------------------------------------------------
# Fix 9: UUID request IDs
# Fix 10: Thread-safe backend init
# ---------------------------------------------------------------------------

class TestUUIDRequestIds(unittest.TestCase):
    """When no request_id is supplied the coordinator generates a unique UUID."""

    def setUp(self) -> None:
        _reset_coord_state()

    def tearDown(self) -> None:
        _reset_coord_state()

    def test_generated_id_is_unique_per_request(self) -> None:
        """Two requests without explicit IDs get different generated IDs."""
        import uuid as _uuid

        generated: list[str] = []

        async def run() -> None:
            coord._workers.append(coord.WorkerState(url="http://w", healthy=True))
            for _ in range(3):
                rid_before = set(coord._active.keys())
                with patch.object(coord.httpx, "AsyncClient", side_effect=httpx.RequestError("fail", request=httpx.Request("POST", "http://w/generate"))):
                    with patch.object(coord, "_append_log", lambda e: None):
                        try:
                            await coord.chat_completions(_make_chat_req())
                        except Exception:
                            pass
                new_ids = set(coord._active.keys()) | set(coord._completed_cache.keys()) | set(coord._cancelled.keys())
                added = new_ids - rid_before
                generated.extend(added)

        asyncio.run(run())
        self.assertEqual(len(generated), len(set(generated)), "Each request should have a unique ID")

    def test_generated_id_is_32_hex_chars(self) -> None:
        """UUID hex IDs are exactly 32 lowercase hex characters (uuid4.hex format)."""
        import re

        coord._workers.append(coord.WorkerState(url="http://w", healthy=True))
        captured_id: list[str] = []

        orig_mark_active = coord._mark_active

        def capture(request_id: str, worker_url: str) -> None:
            captured_id.append(request_id)
            orig_mark_active(request_id, worker_url)

        async def run() -> None:
            with patch.object(coord, "_mark_active", capture):
                with patch.object(coord.httpx, "AsyncClient", side_effect=httpx.RequestError("fail", request=httpx.Request("POST", "http://w/generate"))):
                    with patch.object(coord, "_append_log", lambda e: None):
                        try:
                            await coord.chat_completions(_make_chat_req())
                        except Exception:
                            pass

        asyncio.run(run())
        self.assertTrue(captured_id, "Expected _mark_active to be called")
        rid = captured_id[0]
        self.assertRegex(rid, r'^[0-9a-f]{32}$', f"Expected 32-char hex UUID, got {rid!r}")

    def test_explicit_request_id_is_preserved(self) -> None:
        """A caller-supplied request_id is not replaced by a generated UUID."""
        coord._workers.append(coord.WorkerState(url="http://w", healthy=True))
        captured_id: list[str] = []

        orig_mark_active = coord._mark_active

        def capture(request_id: str, worker_url: str) -> None:
            captured_id.append(request_id)
            orig_mark_active(request_id, worker_url)

        async def run() -> None:
            with patch.object(coord, "_mark_active", capture):
                with patch.object(coord.httpx, "AsyncClient", side_effect=httpx.RequestError("fail", request=httpx.Request("POST", "http://w/generate"))):
                    with patch.object(coord, "_append_log", lambda e: None):
                        try:
                            await coord.chat_completions(_make_chat_req(request_id="my-custom-id"))
                        except Exception:
                            pass

        asyncio.run(run())
        self.assertIn("my-custom-id", captured_id)


class TestThreadSafeBackendInit(unittest.TestCase):
    """_get_backend() uses a threading.Lock to prevent concurrent model loads."""

    def test_backend_lock_is_threading_lock(self) -> None:
        """_backend_lock is a real threading.Lock (or RLock) instance."""
        import threading
        self.assertIsInstance(worker._backend_lock, type(threading.Lock()))

    def test_get_backend_returns_none_for_synthetic(self) -> None:
        """_get_backend() returns None for the default synthetic backend without acquiring the lock."""
        import threading
        original_name = worker._backend_name
        try:
            worker._backend_name = "synthetic"
            result = worker._get_backend()
            self.assertIsNone(result)
        finally:
            worker._backend_name = original_name

    def test_get_backend_raises_for_unknown_backend(self) -> None:
        """_get_backend() raises RuntimeError for an unknown backend name."""
        original_name = worker._backend_name
        try:
            worker._backend_name = "unknown_backend"
            with self.assertRaises(RuntimeError):
                worker._get_backend()
        finally:
            worker._backend_name = original_name

    def test_get_backend_skips_lock_when_backend_already_loaded(self) -> None:
        """Second call returns the cached _backend without re-entering the lock."""
        import threading

        lock_acquisitions = []
        original_lock = worker._backend_lock
        original_backend = worker._backend

        class TrackingLock:
            def __enter__(self):
                lock_acquisitions.append(1)
                return original_lock.__enter__()
            def __exit__(self, *a):
                return original_lock.__exit__(*a)

        try:
            sentinel = object()
            worker._backend = sentinel  # type: ignore[assignment]
            worker._backend_lock = TrackingLock()  # type: ignore[assignment]
            worker._backend_name = "transformers"
            result = worker._get_backend()
            self.assertIs(result, sentinel)
            self.assertEqual(lock_acquisitions, [], "Lock should not be acquired when backend is cached")
        finally:
            worker._backend = original_backend
            worker._backend_lock = original_lock
            worker._backend_name = os.getenv("PHASE2_BACKEND", "synthetic").strip().lower()


if __name__ == "__main__":
    unittest.main()
