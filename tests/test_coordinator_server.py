from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from runtime.phase2 import coordinator_server as coord


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(self, script: list[object], calls: list[tuple[str, str, dict | None]]) -> None:
        self._script = script
        self._calls = calls

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str) -> FakeResponse:
        self._calls.append(("GET", url, None))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]

    async def post(self, url: str, json: dict | None = None, timeout: float | None = None) -> FakeResponse:
        self._calls.append(("POST", url, json))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]


class CoordinatorServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_db_path = coord.DURABLE_DB_PATH
        coord.DURABLE_DB_PATH = str(Path(self._tmpdir.name) / "coordinator_state.db")
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()
        coord._pending_recovery.clear()
        coord._tenant_inflight.clear()
        coord._tenant_limits = {"default": 16}
        coord._tenant_weights = {"default": 1.0}
        coord._admission_limit = 64
        coord._admission_wait_timeout_ms = 200
        coord._default_deadline_ms = 120000

    def tearDown(self) -> None:
        coord.DURABLE_DB_PATH = self._old_db_path
        self._tmpdir.cleanup()
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()
        coord._pending_recovery.clear()
        coord._tenant_inflight.clear()
        coord._ttft_ms_samples.clear()
        coord._metrics.update(
            {
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
        )

    def test_nonstream_retry_uses_second_worker_after_transport_failure(self) -> None:
        coord._workers.extend(
            [
                coord.WorkerState(url="http://worker-a", healthy=True, inflight=0),
                coord.WorkerState(url="http://worker-b", healthy=True, inflight=0),
            ]
        )
        calls: list[tuple[str, str, dict | None]] = []
        fail = httpx.RequestError("boom", request=httpx.Request("POST", "http://worker-a/generate"))
        script = [
            fail,
            FakeResponse({"request_id": "req-1", "text": "from-worker-b", "cancelled": False}),
        ]

        with patch.object(coord.httpx, "AsyncClient", lambda timeout=None: FakeAsyncClient(script, calls)):
            with patch.object(coord, "_append_log", lambda event: None):
                result = asyncio.run(
                    coord.chat_completions(
                        coord.ChatRequest(
                            model="test-model",
                            messages=[coord.Message(role="user", content="hello")],
                            max_tokens=4,
                            temperature=0.0,
                            stream=False,
                            request_id="req-1",
                        )
                    )
                )

        self.assertEqual(result["choices"][0]["message"]["content"], "from-worker-b")
        self.assertFalse(coord._workers[0].healthy)
        self.assertEqual(
            calls,
            [
                ("POST", "http://worker-a/generate", {"request_id": "req-1", "prompt": "hello", "max_tokens": 4, "temperature": 0.0, "deadline_unix_ms": unittest.mock.ANY, "prefill_handoff_id": None}),
                ("POST", "http://worker-b/generate", {"request_id": "req-1", "prompt": "hello", "max_tokens": 4, "temperature": 0.0, "deadline_unix_ms": unittest.mock.ANY, "prefill_handoff_id": None}),
            ],
        )
        first_deadline = calls[0][2]["deadline_unix_ms"] if calls[0][2] else None
        self.assertIsInstance(first_deadline, int)

    def test_cancel_forwards_to_active_worker_and_marks_request(self) -> None:
        coord._mark_active("req-cancel", "http://worker-a")
        calls: list[tuple[str, str, dict | None]] = []
        script = [FakeResponse({})]

        with patch.object(coord.httpx, "AsyncClient", lambda timeout=None: FakeAsyncClient(script, calls)):
            with patch.object(coord, "_append_log", lambda event: None):
                result = asyncio.run(coord.cancel_request("req-cancel"))

        self.assertEqual(result, {"request_id": "req-cancel", "status": "cancel_accepted"})
        self.assertIn("req-cancel", coord._cancelled)
        self.assertEqual(calls, [("POST", "http://worker-a/cancel/req-cancel", None)])

    def test_duplicate_request_id_returns_cached_completion(self) -> None:
        cached = {
            "id": "chatcmpl-phase2",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "cached"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        coord._cache_completion("req-cache", cached)

        with patch.object(coord.httpx, "AsyncClient", side_effect=AssertionError("cached request should not route")):
            result = asyncio.run(
                coord.chat_completions(
                    coord.ChatRequest(
                        model="test-model",
                        messages=[coord.Message(role="user", content="ignored")],
                        max_tokens=4,
                        temperature=0.0,
                        stream=False,
                        request_id="req-cache",
                    )
                )
            )

        self.assertEqual(result, cached)

    def test_load_recovery_state_reconstructs_pending_and_completed_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "requests.jsonl"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps({"event": "admitted", "request_id": "req-a"}),
                        json.dumps({"event": "admitted", "request_id": "req-b"}),
                        json.dumps({"event": "completed", "request_id": "req-a", "response": {"request_id": "req-a", "text": "done"}}),
                        json.dumps({"event": "failed", "request_id": "req-c"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(coord, "REQUEST_LOG", str(log_path)):
                coord._load_recovery_state()

        self.assertEqual(coord._pending_recovery, {"req-b"})
        self.assertEqual(coord._get_completion("req-a"), {"request_id": "req-a", "text": "done"})

    def test_metrics_endpoint_reports_inflight_retries_cancellations_and_ttft(self) -> None:
        coord._metrics["requests_total"] = 3
        coord._metrics["stream_requests_total"] = 2
        coord._metrics["nonstream_requests_total"] = 1
        coord._metrics["retry_attempts_total"] = 4
        coord._metrics["cancellations_total"] = 5
        coord._mark_active("req-1", "http://worker-a")
        coord._workers.extend(
            [
                coord.WorkerState(url="http://worker-a", healthy=True, inflight=2),
                coord.WorkerState(url="http://worker-b", healthy=True, inflight=1),
            ]
        )
        coord._ttft_ms_samples.extend([10.0, 20.0, 30.0])

        result = asyncio.run(coord.metrics())

        self.assertEqual(result["requests_total"], 3)
        self.assertEqual(result["stream_requests_total"], 2)
        self.assertEqual(result["nonstream_requests_total"], 1)
        self.assertEqual(result["retry_attempts_total"], 4)
        self.assertEqual(result["cancellations_total"], 5)
        self.assertEqual(result["admission_rejections_total"], 0)
        self.assertEqual(result["worker_transport_failures_total"], 0)
        self.assertEqual(result["worker_stream_transport_failures_total"], 0)
        self.assertEqual(result["request_timeouts_total"], 0)
        self.assertEqual(result["current_inflight_requests"], 1)
        self.assertEqual(result["workers_inflight_total"], 3)
        self.assertEqual(result["ttft_sample_count"], 3)
        self.assertEqual(result["ttft_ms_p50"], 20.0)

    def test_cancel_is_idempotent_for_same_request_id(self) -> None:
        with patch.object(coord.httpx, "AsyncClient", side_effect=AssertionError("no active worker expected")):
            with patch.object(coord, "_append_log", lambda event: None):
                first = asyncio.run(coord.cancel_request("req-idem"))
                second = asyncio.run(coord.cancel_request("req-idem"))

        self.assertEqual(first, {"request_id": "req-idem", "status": "cancel_accepted"})
        self.assertEqual(second, {"request_id": "req-idem", "status": "cancel_accepted"})
        self.assertIn("req-idem", coord._cancelled)

    def test_cancelled_request_id_is_rejected_without_cached_completion(self) -> None:
        coord._workers.append(coord.WorkerState(url="http://worker-a", healthy=True, inflight=0))
        coord._mark_cancelled("req-cancelled")

        with self.assertRaises(HTTPException) as exc:
            asyncio.run(
                coord.chat_completions(
                    coord.ChatRequest(
                        model="test-model",
                        messages=[coord.Message(role="user", content="hello")],
                        max_tokens=4,
                        temperature=0.0,
                        stream=False,
                        request_id="req-cancelled",
                    )
                )
            )

        self.assertEqual(exc.exception.status_code, 499)

    def test_choose_worker_filters_by_role(self) -> None:
        coord._workers.extend(
            [
                coord.WorkerState(url="http://prefill-a", role="prefill", healthy=True, inflight=0),
                coord.WorkerState(url="http://decode-a", role="decode", healthy=True, inflight=1),
                coord.WorkerState(url="http://decode-b", role="decode", healthy=True, inflight=0),
            ]
        )
        chosen = coord._choose_worker(role="decode", tenant_key="m1")
        self.assertEqual(chosen.url, "http://decode-b")

    def test_nonstream_prefill_decode_routes_to_decode_worker(self) -> None:
        coord._workers.extend(
            [
                coord.WorkerState(url="http://prefill-a", role="prefill", healthy=True, inflight=0),
                coord.WorkerState(url="http://decode-a", role="decode", healthy=True, inflight=0),
            ]
        )
        calls: list[tuple[str, str, dict | None]] = []
        script = [
            FakeResponse({"prefill_handoff_id": "hid-1"}),
            FakeResponse({"request_id": "req-pd", "text": "decoded", "cancelled": False}),
        ]

        with patch.object(coord.httpx, "AsyncClient", lambda timeout=None: FakeAsyncClient(script, calls)):
            with patch.object(coord, "_append_log", lambda event: None):
                result = asyncio.run(
                    coord.chat_completions(
                        coord.ChatRequest(
                            model="test-model",
                            messages=[coord.Message(role="user", content="hello world")],
                            max_tokens=4,
                            temperature=0.0,
                            stream=False,
                            request_id="req-pd",
                        )
                    )
                )

        self.assertEqual(result["choices"][0]["message"]["content"], "decoded")
        self.assertEqual(calls[0][1], "http://prefill-a/prefill")
        self.assertEqual(calls[1][1], "http://decode-a/generate")
        self.assertEqual(calls[1][2]["prefill_handoff_id"], "hid-1")
        self.assertEqual(coord._workers[0].inflight, 0)
        self.assertEqual(coord._workers[1].inflight, 0)

    def test_metrics_exposes_scheduler_and_autoscale_fields(self) -> None:
        result = asyncio.run(coord.metrics())
        self.assertIn("scheduler_policy", result)
        self.assertIn("autoscale_admission_limit", result)
        self.assertIn("tenant_limits", result)
        self.assertIn("tenant_weights", result)

    def test_admission_rejects_when_tenant_limit_reached(self) -> None:
        coord._workers.append(coord.WorkerState(url="http://worker-a", role="decode", healthy=True, inflight=0))
        coord._tenant_limits = {"default": 1}
        coord._tenant_inflight["default"] = 1
        with self.assertRaises(HTTPException) as exc:
            asyncio.run(
                coord.chat_completions(
                    coord.ChatRequest(
                        model="test-model",
                        messages=[coord.Message(role="user", content="hello")],
                        request_id="req-tenant-limit",
                    )
                )
            )
        self.assertEqual(exc.exception.status_code, 429)

    def test_admission_waits_for_capacity_instead_of_immediate_reject(self) -> None:
        coord._workers.append(coord.WorkerState(url="http://worker-a", role="decode", healthy=True, inflight=0))
        coord._admission_limit = 1
        coord._admission_wait_timeout_ms = 250
        coord._mark_active("existing", "http://worker-a")
        calls: list[tuple[str, str, dict | None]] = []
        script = [FakeResponse({"request_id": "req-q", "text": "ok", "cancelled": False})]

        async def clear_slot() -> None:
            await asyncio.sleep(0.05)
            coord._active.pop("existing", None)

        async def run_test() -> dict[str, object]:
            asyncio.create_task(clear_slot())
            with patch.object(coord.httpx, "AsyncClient", lambda timeout=None: FakeAsyncClient(script, calls)):
                with patch.object(coord, "_append_log", lambda event: None):
                    return await coord.chat_completions(
                        coord.ChatRequest(
                            model="test-model",
                            messages=[coord.Message(role="user", content="hello")],
                            max_tokens=4,
                            temperature=0.0,
                            stream=False,
                            request_id="req-q",
                        )
                    )

        result = asyncio.run(run_test())
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.assertEqual(coord._metrics["admission_rejections_total"], 0)

    def test_load_recovery_state_reads_durable_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "coord.db"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "CREATE TABLE request_state (request_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, status TEXT NOT NULL, updated_unix_ms INTEGER NOT NULL, response_json TEXT)"
                )
                conn.execute(
                    "INSERT INTO request_state VALUES (?, ?, ?, ?, ?)",
                    ("req-db-done", "default", "completed", int(time.time() * 1000), json.dumps({"request_id": "req-db-done", "text": "done"})),
                )
                conn.execute(
                    "INSERT INTO request_state VALUES (?, ?, ?, ?, ?)",
                    ("req-db-pending", "default", "admitted", int(time.time() * 1000), None),
                )
                conn.commit()
            finally:
                conn.close()
            with patch.object(coord, "DURABLE_DB_PATH", str(db_path)):
                coord._load_recovery_state()

        self.assertEqual(coord._pending_recovery, {"req-db-pending"})
        self.assertEqual(coord._get_completion("req-db-done"), {"request_id": "req-db-done", "text": "done"})


if __name__ == "__main__":
    unittest.main()
