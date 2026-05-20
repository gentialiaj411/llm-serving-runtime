from __future__ import annotations

import asyncio
import json
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
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()
        coord._pending_recovery.clear()

    def tearDown(self) -> None:
        coord._workers.clear()
        coord._active.clear()
        coord._completed_cache.clear()
        coord._cancelled.clear()
        coord._pending_recovery.clear()
        coord._ttft_ms_samples.clear()
        coord._metrics.update(
            {
                "requests_total": 0,
                "stream_requests_total": 0,
                "nonstream_requests_total": 0,
                "retry_attempts_total": 0,
                "cancellations_total": 0,
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
                ("POST", "http://worker-a/generate", {"request_id": "req-1", "prompt": "hello", "max_tokens": 4, "temperature": 0.0, "deadline_unix_ms": None}),
                ("POST", "http://worker-b/generate", {"request_id": "req-1", "prompt": "hello", "max_tokens": 4, "temperature": 0.0, "deadline_unix_ms": None}),
            ],
        )

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


if __name__ == "__main__":
    unittest.main()
