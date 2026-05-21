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


def _collect_tokens(speculative: bool) -> list[str]:
    port = _free_port()
    env = os.environ.copy()
    env["PHASE2_BACKEND"] = "transformers"
    env["PHASE2_SPECULATIVE"] = "1" if speculative else "0"
    env["PHASE2_SPEC_K"] = "4"
    env["HF_MODEL_ID"] = "sshleifer/tiny-gpt2"
    env["HF_DRAFT_MODEL_ID"] = "sshleifer/tiny-gpt2"
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
        deadline = time.time() + 30.0
        while time.time() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.25)
        else:
            raise AssertionError("worker did not become healthy")

        body = {
            "request_id": f"spec-parity-{int(speculative)}-{time.time_ns()}",
            "prompt": "The quick brown fox",
            "max_tokens": 20,
            "temperature": 0.0,
        }
        events: list[dict] = []
        with httpx.stream("POST", f"http://127.0.0.1:{port}/generate_stream", json=body, timeout=120.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    events.append(json.loads(line))
        return [str(e.get("text", e.get("token", ""))) for e in events if e.get("type") == "token"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


class SpeculativeParityTests(unittest.TestCase):
    def test_speculative_matches_greedy_output(self) -> None:
        try:
            import pytest
        except ImportError:
            self.skipTest("pytest is required for importorskip")

        try:
            pytest.importorskip("torch")
            pytest.importorskip("transformers")
        except pytest.skip.Exception as exc:
            self.skipTest(str(exc))

        self.assertEqual(_collect_tokens(False), _collect_tokens(True))


if __name__ == "__main__":
    unittest.main()
