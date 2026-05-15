from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient


class PhaseServerTests(unittest.TestCase):
    def test_phase1_nonstream_and_stream_shapes(self) -> None:
        from frontend.phase1_server import app

        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "alpha beta"}],
            "max_tokens": 3,
            "temperature": 0.0,
        }
        with TestClient(app) as client:
            nonstream = client.post("/v1/chat/completions", json={**body, "stream": False})
            self.assertEqual(nonstream.status_code, 200)
            self.assertEqual(nonstream.json()["usage"]["completion_tokens"], 3)

            with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as resp:
                self.assertEqual(resp.status_code, 200)
                data_lines = [
                    line.removeprefix("data: ").strip()
                    for line in resp.iter_lines()
                    if line.startswith("data: ")
                ]

        token_chunks = [json.loads(line) for line in data_lines if line != "[DONE]"]
        contents = [
            chunk["choices"][0]["delta"]["content"]
            for chunk in token_chunks
            if chunk["choices"][0]["delta"].get("content")
        ]
        self.assertEqual("".join(contents), "alpha beta alpha")
        self.assertEqual(data_lines[-1], "[DONE]")

    def test_worker_stream_emits_decode_tokens(self) -> None:
        from runtime.phase2.worker_server import app

        body = {
            "request_id": "test-worker-stream",
            "prompt": "one two",
            "max_tokens": 3,
            "temperature": 0.0,
        }
        with TestClient(app) as client:
            with client.stream("POST", "/generate_stream", json=body) as resp:
                self.assertEqual(resp.status_code, 200)
                events = [json.loads(line) for line in resp.iter_lines() if line]

        self.assertEqual([e["type"] for e in events], ["token", "token", "token", "done"])
        self.assertEqual([e["token"] for e in events[:-1]], ["one", "two", "one"])


if __name__ == "__main__":
    unittest.main()
