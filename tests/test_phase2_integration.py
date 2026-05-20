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


class Phase2IntegrationTests(unittest.TestCase):
    def test_two_workers_coordinator_streaming(self) -> None:
        worker_port_1 = _free_port()
        worker_port_2 = _free_port()
        coord_port = _free_port()

        procs: list[subprocess.Popen[str]] = []
        try:
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "runtime.phase2.worker_server:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(worker_port_1),
                        "--log-level",
                        "warning",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            )
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "runtime.phase2.worker_server:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(worker_port_2),
                        "--log-level",
                        "warning",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            )

            env = os.environ.copy()
            env["WORKER_URLS"] = f"http://127.0.0.1:{worker_port_1},http://127.0.0.1:{worker_port_2}"
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "runtime.phase2.coordinator_server:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(coord_port),
                        "--log-level",
                        "warning",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    env=env,
                )
            )

            deadline = time.time() + 30.0
            health_url = f"http://127.0.0.1:{coord_port}/healthz"
            while time.time() < deadline:
                try:
                    r = httpx.get(health_url, timeout=1.0)
                    if r.status_code == 200 and r.json().get("workers_healthy", 0) >= 2:
                        break
                except Exception:
                    pass
                time.sleep(0.25)
            else:
                self.fail("coordinator did not become healthy with two workers")

            body = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "red blue"}],
                "max_tokens": 3,
                "temperature": 0.0,
                "stream": True,
                "request_id": f"integration-stream-{time.time_ns()}",
            }
            url = f"http://127.0.0.1:{coord_port}/v1/chat/completions"
            with httpx.stream("POST", url, json=body, timeout=20.0) as resp:
                self.assertEqual(resp.status_code, 200)
                raw_lines = [line.strip() for line in resp.iter_lines() if line and line.strip()]
                data_lines = []
                for line in raw_lines:
                    if line.startswith("data:"):
                        data_lines.append(line.split("data:", 1)[1].strip())

            self.assertTrue(data_lines, f"stream contained no data lines; raw={raw_lines}")
            self.assertEqual(data_lines[-1], "[DONE]")
            chunks = [json.loads(line) for line in data_lines[:-1]]
            contents = [
                chunk["choices"][0]["delta"]["content"]
                for chunk in chunks
                if chunk["choices"][0]["delta"].get("content")
            ]
            self.assertEqual("".join(contents), "red blue red")
        finally:
            for p in procs:
                p.terminate()
            for p in procs:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()


if __name__ == "__main__":
    unittest.main()
