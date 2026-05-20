from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import unittest

import httpx


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Phase2RealModelTests(unittest.TestCase):
    def test_worker_stream_uses_transformers_backend(self) -> None:
        try:
            import pytest
        except ImportError:
            self.skipTest("pytest is required for importorskip")

        try:
            pytest.importorskip("torch")
            pytest.importorskip("transformers")
        except pytest.skip.Exception as exc:
            self.skipTest(str(exc))

        port = _free_port()
        env = os.environ.copy()
        env["PHASE2_BACKEND"] = "transformers"
        env["HF_MODEL_ID"] = "sshleifer/tiny-gpt2"
        env["HF_TORCH_DTYPE"] = "float32"

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
        try:
            health_url = f"http://127.0.0.1:{port}/healthz"
            deadline = time.time() + 30.0
            while time.time() < deadline:
                try:
                    r = httpx.get(health_url, timeout=1.0)
                    if r.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(0.25)
            else:
                self.fail("phase2 transformers worker did not become healthy")

            body = {
                "request_id": f"real-model-{time.time_ns()}",
                "prompt": "alpha beta",
                "max_tokens": 8,
                "temperature": 0.0,
            }
            events: list[dict] = []
            with httpx.stream("POST", f"http://127.0.0.1:{port}/generate_stream", json=body, timeout=120.0) as resp:
                self.assertEqual(resp.status_code, 200)
                for line in resp.iter_lines():
                    if line:
                        events.append(json.loads(line))

            token_events = [e for e in events if e.get("type") == "token"]
            self.assertGreaterEqual(len(token_events), 1)
            self.assertEqual(events[-1].get("type"), "done")

            produced = [str(e.get("text", e.get("token", ""))) for e in token_events]
            synthetic = ["alpha", "beta"] * 4
            self.assertNotEqual(produced, synthetic[: len(produced)])
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
