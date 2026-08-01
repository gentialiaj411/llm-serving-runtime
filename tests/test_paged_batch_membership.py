"""Unit test: paged batch-cache split must target the owning decode rows only."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock


class FakeBatchCache:
    def __init__(self, n_rows: int) -> None:
        self.block_ids_rows = [[i] for i in range(n_rows)]
        self._batch_size = n_rows
        self.extracted: list[int] = []

    def extract_cache(self, index: int) -> SimpleNamespace:
        self.block_ids_rows[index]  # raises IndexError if mismatch
        self.extracted.append(index)
        return SimpleNamespace(row=index)


class PagedBatchMembershipTests(unittest.TestCase):
    def test_split_uses_decode_owners_not_all_prefilled(self) -> None:
        """Regression: mixing just-prefilled rows into the split list used to IndexError."""
        from runtime.phase2.worker_server import TransformersBackend

        backend = TransformersBackend.__new__(TransformersBackend)
        backend.kv_backend = "paged"
        backend.kv_pool = object()

        # Two already-decoding rows + one just-prefilled row in the same scheduler batch.
        decode_a = SimpleNamespace(
            req=SimpleNamespace(request_id="a", adapter="base"),
            prompt_prefilled=True,
            past_key_values=SimpleNamespace(),
            last_token_id=1,
            generated_token_ids=[],
            input_ids=None,
            attention_mask=None,
        )
        decode_b = SimpleNamespace(
            req=SimpleNamespace(request_id="b", adapter="base"),
            prompt_prefilled=True,
            past_key_values=SimpleNamespace(),
            last_token_id=2,
            generated_token_ids=[],
            input_ids=None,
            attention_mask=None,
        )
        fresh = SimpleNamespace(
            req=SimpleNamespace(request_id="c", adapter="base"),
            prompt_prefilled=False,
            past_key_values=SimpleNamespace(),
            last_token_id=None,
            generated_token_ids=[],
            input_ids=MagicMock(shape=(1, 4)),
            attention_mask=None,
        )

        live = FakeBatchCache(2)

        def fake_prefill(group: list[Any]) -> dict[str, str]:
            for state in group:
                state.prompt_prefilled = True
                state.last_token_id = 9
                state.generated_token_ids = [9]
            return {state.req.request_id: "x" for state in group}

        def fake_decode(decode_states: list[Any], batch_cache: Any) -> tuple[dict[str, str], Any]:
            self.assertEqual([s.req.request_id for s in decode_states], ["a", "b"])
            self.assertIs(batch_cache, None)
            for state in decode_states:
                state.generated_token_ids.append(7)
                state.last_token_id = 7
            return {state.req.request_id: "y" for state in decode_states}, live

        backend._paged_prefill_groups = fake_prefill  # type: ignore[method-assign]
        backend._paged_decode_step = fake_decode  # type: ignore[method-assign]
        backend.set_active_adapter = lambda *_a, **_k: None  # type: ignore[method-assign]

        emitted = TransformersBackend._paged_next_tokens_batch_for_adapter(
            backend, [decode_a, decode_b, fresh], max_tokens=1
        )

        self.assertEqual(live.extracted, [0, 1])
        self.assertEqual(emitted["a"], ["y"])
        self.assertEqual(emitted["b"], ["y"])
        self.assertEqual(emitted["c"], ["x"])
        self.assertIsInstance(decode_a.past_key_values, SimpleNamespace)
        self.assertEqual(decode_a.past_key_values.row, 0)
        self.assertEqual(decode_b.past_key_values.row, 1)


if __name__ == "__main__":
    unittest.main()
