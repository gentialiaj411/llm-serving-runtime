from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import unittest
import asyncio
from unittest.mock import patch

import httpx


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Phase2WorkerLiveKVTests(unittest.TestCase):
    def _start_worker(self, extra_env: dict[str, str] | None = None) -> tuple[subprocess.Popen[str], int]:
        port = _free_port()
        env = os.environ.copy()
        env.setdefault("KV_TOTAL_BLOCKS", "128")
        env.setdefault("KV_BLOCK_SIZE_TOKENS", "4")
        if extra_env:
            env.update(extra_env)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "runtime.phase2.worker_server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        health_url = f"http://127.0.0.1:{port}/healthz"
        deadline = time.time() + 30.0
        while time.time() < deadline:
            try:
                if httpx.get(health_url, timeout=1.0).status_code == 200:
                    return proc, port
            except Exception:
                pass
            time.sleep(0.25)
        proc.terminate()
        proc.wait(timeout=10)
        self.fail("worker did not become healthy")

    def _stop(self, proc: subprocess.Popen[str]) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    def test_kv_alloc_and_free_on_completion_and_metrics_peak(self) -> None:
        proc, port = self._start_worker()
        try:
            body = {"request_id": f"done-{time.time_ns()}", "prompt": "a b c", "max_tokens": 6, "temperature": 0.0}
            resp = httpx.post(f"http://127.0.0.1:{port}/generate", json=body, timeout=20.0)
            self.assertEqual(resp.status_code, 200)
            metrics = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5.0).json()
            self.assertEqual(metrics["active_allocations"], 0)
            self.assertEqual(metrics["used_blocks"], 0)
            self.assertGreater(metrics["peak_kv_bytes"], 0)
            self.assertGreater(metrics["allocations_total"], 0)
            self.assertGreater(metrics["frees_total"], 0)
        finally:
            self._stop(proc)

    def test_kv_freed_on_cancel(self) -> None:
        proc, port = self._start_worker()
        rid = f"cancel-{time.time_ns()}"
        try:
            with httpx.stream(
                "POST",
                f"http://127.0.0.1:{port}/generate_stream",
                json={"request_id": rid, "prompt": "x y z", "max_tokens": 2048, "temperature": 0.0},
                timeout=20.0,
            ) as resp:
                self.assertEqual(resp.status_code, 200)
                line_iter = resp.iter_lines()
                first = next(line for line in line_iter if line)
                self.assertIn('"type": "token"', first)
                cancel = httpx.post(f"http://127.0.0.1:{port}/cancel/{rid}", timeout=5.0)
                self.assertEqual(cancel.status_code, 200)
                events = [json.loads(line) for line in line_iter if line]
            self.assertTrue(any(e.get("type") == "cancelled" for e in events))

            deadline = time.time() + 5.0
            while time.time() < deadline:
                metrics = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5.0).json()
                if metrics["active_allocations"] == 0 and metrics["used_blocks"] == 0:
                    break
                time.sleep(0.05)
            self.assertEqual(metrics["frees_cancel_total"], 1)
        finally:
            self._stop(proc)

    def test_kv_freed_on_timeout(self) -> None:
        proc, port = self._start_worker()
        try:
            deadline_ms = int(time.time() * 1000) + 30
            resp = httpx.post(
                f"http://127.0.0.1:{port}/generate",
                json={
                    "request_id": f"timeout-{time.time_ns()}",
                    "prompt": "alpha",
                    "max_tokens": 64,
                    "temperature": 0.0,
                    "deadline_unix_ms": deadline_ms,
                },
                timeout=20.0,
            )
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json().get("timed_out"))
            metrics = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5.0).json()
            self.assertEqual(metrics["active_allocations"], 0)
            self.assertEqual(metrics["frees_timeout_total"], 1)
        finally:
            self._stop(proc)

    def test_transformers_path_continuous_batches_multiple_concurrency(self) -> None:
        if os.getenv("PHASE2_REAL_MODEL_TESTS", "0") != "1":
            self.skipTest("set PHASE2_REAL_MODEL_TESTS=1 to run real-model batching smoke")
        try:
            import pytest
        except ImportError:
            self.skipTest("pytest is required for importorskip")
        try:
            pytest.importorskip("torch")
            pytest.importorskip("transformers")
        except pytest.skip.Exception as exc:
            self.skipTest(str(exc))

        proc, port = self._start_worker(
            {
                "PHASE2_BACKEND": "transformers",
                "HF_MODEL_ID": "sshleifer/tiny-gpt2",
                "HF_TORCH_DTYPE": "float32",
                "PHASE2_BATCH_DECODE_STEPS": "1",
            }
        )
        try:
            client = httpx.Client(timeout=120.0)
            reqs = []
            for i in range(2):
                reqs.append(
                    client.build_request(
                        "POST",
                        f"http://127.0.0.1:{port}/generate",
                        json={"request_id": f"batch-{i}-{time.time_ns()}", "prompt": "hello world", "max_tokens": 4, "temperature": 0.0},
                    )
                )
            responses = [client.send(r) for r in reqs]
            for r in responses:
                self.assertEqual(r.status_code, 200)
                self.assertFalse(r.json().get("cancelled"))
            metrics = client.get(f"http://127.0.0.1:{port}/metrics").json()
            self.assertGreaterEqual(metrics["continuous_batches_total"], 1)
            self.assertGreaterEqual(metrics["batched_requests_total"], 2)
        finally:
            self._stop(proc)

    def test_kv_freed_on_backend_error(self) -> None:
        from fastapi.testclient import TestClient
        from runtime.phase2 import worker_server as worker

        class BrokenBackend:
            def init_state(self, state: object) -> None:
                raise RuntimeError("intentional init failure")

        with patch.object(worker, "_backend_name", "synthetic"):
            with patch.object(worker, "_get_backend", return_value=BrokenBackend()):
                with TestClient(worker.app) as client:
                    resp = client.post(
                        "/generate",
                        json={
                            "request_id": f"err-{time.time_ns()}",
                            "prompt": "hello",
                            "max_tokens": 3,
                            "temperature": 0.0,
                        },
                    )
                    self.assertEqual(resp.status_code, 200)
                    payload = resp.json()
                    self.assertIn("error", payload)
                    metrics = client.get("/metrics").json()
                    self.assertEqual(metrics["active_allocations"], 0)
                    self.assertGreaterEqual(metrics["frees_error_total"], 1)

    def test_concurrent_stream_metrics_peak_and_batch(self) -> None:
        proc, port = self._start_worker({"PHASE2_DECODE_STEP_MS": "8"})
        try:
            async def run_clients() -> None:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async def one(i: int) -> dict:
                        rid = f"conc-{i}-{time.time_ns()}"
                        events = []
                        async with client.stream(
                            "POST",
                            f"http://127.0.0.1:{port}/generate_stream",
                            json={"request_id": rid, "prompt": "one two three", "max_tokens": 32, "temperature": 0.0},
                        ) as resp:
                            self.assertEqual(resp.status_code, 200)
                            async for line in resp.aiter_lines():
                                if line:
                                    event = json.loads(line)
                                    events.append(event)
                                    if event.get("type") in {"done", "cancelled", "timed_out", "error"}:
                                        break
                        return {"rid": rid, "events": events}

                    results = await asyncio.gather(*[one(i) for i in range(4)])
                    for result in results:
                        self.assertEqual(result["events"][-1]["type"], "done")

                    metrics = (await client.get(f"http://127.0.0.1:{port}/metrics")).json()
                    self.assertGreaterEqual(metrics["peak_active_requests"], 4)
                    self.assertGreaterEqual(metrics["max_batch_size"], 2)
                    self.assertEqual(metrics["active_allocations"], 0)

            asyncio.run(run_clients())
        finally:
            self._stop(proc)


if __name__ == "__main__":
    unittest.main()
