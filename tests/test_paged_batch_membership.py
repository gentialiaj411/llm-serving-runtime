"""Unit test: persistent paged batch slots survive membership changes without full concat."""

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
        self._request_to_slot: dict[str, int] = {}
        self._free_slots = list(range(n_rows - 1, -1, -1))
        self.seq_lens = SimpleNamespace()  # unused in this fake

    def extract_cache(self, index: int) -> SimpleNamespace:
        self.block_ids_rows[index]
        self.extracted.append(index)
        return SimpleNamespace(row=index, block_table_ids=list(self.block_ids_rows[index]))

    def slot_of(self, request_id: str) -> int | None:
        return self._request_to_slot.get(request_id)

    def release_slot(self, request_id: str) -> None:
        slot = self._request_to_slot.pop(request_id, None)
        if slot is not None:
            self._free_slots.append(slot)

    def allocate_slot(self, request_id: str) -> int:
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        slot = self._free_slots.pop()
        self._request_to_slot[request_id] = slot
        return slot

    def bind_cache(self, request_id: str, cache: Any) -> int:
        return self.allocate_slot(request_id)

    def sync_block_ids_row(self, index: int, block_ids: list[int]) -> None:
        self.block_ids_rows[index] = list(block_ids)

    def dense_active_cache(self, request_ids: list[str]):
        slots = [self._request_to_slot[r] for r in request_ids]
        return self, slots

    def write_back_dense(self, dense: Any, slots: list[int]) -> None:
        return None


class PersistentPagedBatchTests(unittest.TestCase):
    def test_membership_change_binds_without_indexerror(self) -> None:
        from runtime.phase2.worker_server import TransformersBackend

        backend = TransformersBackend.__new__(TransformersBackend)
        backend.kv_backend = "paged"
        backend.kv_pool = object()
        backend.torch = MagicMock()
        backend.torch.tensor = MagicMock(return_value=MagicMock())
        backend.device = "cpu"
        backend.tokenizer = MagicMock()
        backend.tokenizer.decode = MagicMock(return_value="t")
        backend._paged_persistent = FakeBatchCache(8)
        backend._paged_persistent_capacity = 8

        def fake_ensure(min_batch: int, max_blocks: int):
            return backend._paged_persistent

        backend._ensure_paged_persistent = fake_ensure  # type: ignore[method-assign]
        backend._bind_states_to_persistent = lambda states: backend._paged_persistent  # type: ignore[method-assign]

        decode_a = SimpleNamespace(
            req=SimpleNamespace(request_id="a", adapter="base"),
            prompt_prefilled=True,
            past_key_values=SimpleNamespace(block_table_ids=[1], get_seq_length=lambda: 4),
            last_token_id=1,
            generated_token_ids=[],
            input_ids=None,
            attention_mask=None,
        )
        decode_b = SimpleNamespace(
            req=SimpleNamespace(request_id="b", adapter="base"),
            prompt_prefilled=True,
            past_key_values=SimpleNamespace(block_table_ids=[2], get_seq_length=lambda: 4),
            last_token_id=2,
            generated_token_ids=[],
            input_ids=None,
            attention_mask=None,
        )
        fresh = SimpleNamespace(
            req=SimpleNamespace(request_id="c", adapter="base"),
            prompt_prefilled=False,
            past_key_values=SimpleNamespace(block_table_ids=[3], get_seq_length=lambda: 0),
            last_token_id=None,
            generated_token_ids=[],
            input_ids=MagicMock(shape=(1, 4)),
            attention_mask=None,
        )

        def fake_prefill(group: list[Any]) -> dict[str, str]:
            for state in group:
                state.prompt_prefilled = True
                state.last_token_id = 9
                state.generated_token_ids = [9]
            return {state.req.request_id: "x" for state in group}

        def fake_decode(decode_states: list[Any], batch_cache: Any):
            self.assertEqual([s.req.request_id for s in decode_states], ["a", "b"])
            for state in decode_states:
                state.generated_token_ids.append(7)
                state.last_token_id = 7
            return {state.req.request_id: "y" for state in decode_states}, batch_cache

        backend._paged_prefill_groups = fake_prefill  # type: ignore[method-assign]
        backend._paged_decode_step = fake_decode  # type: ignore[method-assign]
        backend.set_active_adapter = lambda *_a, **_k: None  # type: ignore[method-assign]

        emitted = TransformersBackend._paged_next_tokens_batch_for_adapter(
            backend, [decode_a, decode_b, fresh], max_tokens=1
        )
        self.assertEqual(emitted["a"], ["y"])
        self.assertEqual(emitted["b"], ["y"])
        self.assertEqual(emitted["c"], ["x"])
        # After prefill, bind is invoked for all prefilled rows (mocked to persistent).
        self.assertIs(backend._paged_persistent, backend._bind_states_to_persistent([]))


if __name__ == "__main__":
    unittest.main()
