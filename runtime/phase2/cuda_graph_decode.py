"""CUDA-graph capture/replay for steady-state paged decode (PHASE2_CUDA_GRAPH=1)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

from runtime.phase2.paged_kv_cache import BlockPagedBatchCache, BlockPagedBatchLayer


def cuda_graph_enabled() -> bool:
    return os.getenv("PHASE2_CUDA_GRAPH", "0").strip() == "1"


def cuda_graph_try_hf() -> bool:
    """Opt-in: attempting CUDAGraph around full HF+paged forward often invalidates the stream."""
    return os.getenv("PHASE2_CUDA_GRAPH_TRY_HF", "0").strip() == "1"


@dataclass
class CudaGraphBucket:
    batch_size: int
    max_blocks: int
    static_input_ids: torch.Tensor
    dense_cache: BlockPagedBatchCache
    static_cache_position: torch.Tensor | None = None
    graph: torch.cuda.CUDAGraph | None = None
    static_logits: torch.Tensor | None = None
    warmups_done: int = 0
    capture_failed: bool = False
    capture_error: str = ""
    replays: int = 0


@dataclass
class CudaGraphDecodeManager:
    """Bucketed CUDA-graph decode keyed by (batch_size, max_blocks).

    Full HuggingFace Qwen2 + paged-KV capture often fails with
    ``cudaErrorStreamCaptureInvalidated`` (host syncs inside transformers).
    On failure the manager disables itself and callers fall back to eager.
    """

    model: Any
    device: torch.device
    warmup_steps: int = 3
    buckets: dict[tuple[int, int], CudaGraphBucket] = field(default_factory=dict)
    disabled: bool = False
    disable_reason: str = ""

    def _bucket_key(self, batch_size: int, max_blocks: int) -> tuple[int, int]:
        quantized = 1
        while quantized < max_blocks:
            quantized *= 2
        return batch_size, quantized

    def _ensure_bucket(
        self,
        batch_size: int,
        max_blocks: int,
        template: BlockPagedBatchCache,
    ) -> CudaGraphBucket:
        key = self._bucket_key(batch_size, max_blocks)
        bucket = self.buckets.get(key)
        if bucket is not None:
            return bucket

        dense = BlockPagedBatchCache.empty(
            template._config,
            template.pool,
            max_batch_size=batch_size,
            max_blocks=key[1],
        )
        dense._graph_safe = True  # type: ignore[attr-defined]
        for i in range(batch_size):
            dense._slot_occupied[i] = True
        dense._free_slots = []
        for layer in dense.layers:
            assert isinstance(layer, BlockPagedBatchLayer)
            layer._row_seq_lens = dense.seq_lens.clone()

        bucket = CudaGraphBucket(
            batch_size=batch_size,
            max_blocks=key[1],
            static_input_ids=torch.zeros(batch_size, 1, device=self.device, dtype=torch.long),
            dense_cache=dense,
            static_cache_position=torch.zeros(batch_size, device=self.device, dtype=torch.long),
        )
        self.buckets[key] = bucket
        return bucket

    def _copy_rows_into_bucket(
        self,
        bucket: CudaGraphBucket,
        source: BlockPagedBatchCache,
        slots: list[int],
        input_ids: torch.Tensor,
    ) -> None:
        dense = bucket.dense_cache
        dense.ensure_max_blocks(bucket.max_blocks)
        for i, slot in enumerate(slots):
            dense.block_tables[i].zero_()
            src_row = source.block_tables[slot]
            n = min(int(src_row.numel()), int(dense.block_tables.shape[1]))
            dense.block_tables[i, :n].copy_(src_row[:n])
            dense.block_ids_rows[i] = list(source.block_ids_rows[slot])
            dense.seq_lens[i] = source.seq_lens[slot]
        for layer in dense.layers:
            assert isinstance(layer, BlockPagedBatchLayer)
            layer._row_seq_lens.copy_(dense.seq_lens)
            layer.is_initialized = True
        dense._cached_seq_len_py = int(dense.seq_lens.max().item())  # type: ignore[attr-defined]
        bucket.static_input_ids.copy_(input_ids)
        assert bucket.static_cache_position is not None
        bucket.static_cache_position.copy_(dense.seq_lens.to(dtype=torch.long))

    def _write_back(
        self,
        persistent: BlockPagedBatchCache,
        bucket: CudaGraphBucket,
        slots: list[int],
        decode_states: list[Any],
    ) -> None:
        for i, slot in enumerate(slots):
            persistent.seq_lens[slot] = bucket.dense_cache.seq_lens[i]
            persistent.sync_block_ids_row(slot, list(bucket.dense_cache.block_ids_rows[i]))
            for layer in persistent.layers:
                if isinstance(layer, BlockPagedBatchLayer):
                    layer._row_seq_lens[slot] = int(bucket.dense_cache.seq_lens[i].item())
        for idx, state in enumerate(decode_states):
            state.past_key_values = persistent.extract_cache(slots[idx])

    def try_decode(
        self,
        *,
        persistent: BlockPagedBatchCache,
        decode_states: list[Any],
        input_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Replay a captured graph; return logits [B, vocab] or None for eager fallback."""
        if self.disabled or not torch.cuda.is_available() or not cuda_graph_enabled():
            return None
        # Default: do not attempt HF capture (poisons the CUDA context on failure).
        # Set PHASE2_CUDA_GRAPH_TRY_HF=1 to experiment; linear smoke still proves CUDAGraph.
        if not cuda_graph_try_hf():
            return None
        batch_size = len(decode_states)
        if batch_size <= 0:
            return None

        request_ids = [state.req.request_id for state in decode_states]
        try:
            slots = [persistent._request_to_slot[rid] for rid in request_ids]
        except KeyError:
            return None

        max_blocks = max(max(len(persistent.block_ids_rows[s]), 1) for s in slots)
        bucket = self._ensure_bucket(batch_size, max_blocks, persistent)
        if bucket.capture_failed:
            return None

        self._copy_rows_into_bucket(bucket, persistent, slots, input_ids)

        if bucket.graph is None:
            logits = self._capture(bucket)
            if logits is None:
                return None
        else:
            bucket.graph.replay()
            bucket.replays += 1
            assert bucket.static_logits is not None
            logits = bucket.static_logits

        self._write_back(persistent, bucket, slots, decode_states)
        return logits

    def _forward(self, bucket: CudaGraphBucket) -> Any:
        return self.model(
            input_ids=bucket.static_input_ids,
            past_key_values=bucket.dense_cache,
            cache_position=bucket.static_cache_position,
            use_cache=True,
        )

    def _capture(self, bucket: CudaGraphBucket) -> torch.Tensor | None:
        """Warm up / capture. Returns logits when this call already ran the model once."""
        try:
            if bucket.warmups_done < self.warmup_steps:
                with torch.no_grad():
                    out = self._forward(bucket)
                bucket.warmups_done += 1
                return out.logits[:, -1, :]

            dense = bucket.dense_cache
            seq_before = dense.seq_lens.clone()
            row_before = [
                layer._row_seq_lens.clone()
                for layer in dense.layers
                if isinstance(layer, BlockPagedBatchLayer)
            ]
            cached_py = getattr(dense, "_cached_seq_len_py", 0)

            # Simple same-stream capture (side-stream warmup invalidated Triton/HF stacks).
            torch.cuda.synchronize()
            for _ in range(2):
                with torch.no_grad():
                    self._forward(bucket)

            # Restore before the real captured step so replay advances one token.
            dense.seq_lens.copy_(seq_before)
            for layer, snap in zip(
                [layer for layer in dense.layers if isinstance(layer, BlockPagedBatchLayer)],
                row_before,
            ):
                layer._row_seq_lens.copy_(snap)
            dense._cached_seq_len_py = cached_py  # type: ignore[attr-defined]
            assert bucket.static_cache_position is not None
            bucket.static_cache_position.copy_(dense.seq_lens.to(dtype=torch.long))

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._forward(bucket)
            bucket.graph = graph
            bucket.static_logits = out.logits[:, -1, :]
            bucket.replays += 1
            return bucket.static_logits
        except Exception as exc:  # noqa: BLE001
            bucket.capture_failed = True
            bucket.capture_error = f"capture failed: {exc}"
            bucket.graph = None
            self.disabled = True
            self.disable_reason = bucket.capture_error
            # Failed capture can leave a sticky CUDA error; clear it for eager fallback.
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            try:
                torch.cuda.cudart().cudaGetLastError()
            except Exception:  # noqa: BLE001
                pass
            return None

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": cuda_graph_enabled(),
            "try_hf": cuda_graph_try_hf(),
            "disabled": self.disabled,
            "disable_reason": self.disable_reason,
            "buckets": [
                {
                    "batch_size": b.batch_size,
                    "max_blocks": b.max_blocks,
                    "captured": b.graph is not None,
                    "capture_failed": b.capture_failed,
                    "capture_error": b.capture_error,
                    "warmups_done": b.warmups_done,
                    "replays": b.replays,
                }
                for b in self.buckets.values()
            ],
        }


def capture_replay_linear_smoke(device: str = "cuda") -> dict[str, Any]:
    """Prove ``torch.cuda.CUDAGraph`` works on this host with a tiny static module."""
    if not torch.cuda.is_available():
        return {"ok": False, "error": "cuda unavailable"}

    torch.manual_seed(0)
    module = torch.nn.Linear(64, 64, bias=False).to(device=device, dtype=torch.float16).eval()
    static_in = torch.randn(4, 64, device=device, dtype=torch.float16)
    for _ in range(3):
        module(static_in)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = module(static_in)

    probe = torch.randn(4, 64, device=device, dtype=torch.float16)
    static_in.copy_(probe)
    graph.replay()
    torch.cuda.synchronize()
    eager = module(probe)
    max_abs = float((static_out - eager).abs().max().item())
    return {"ok": max_abs < 1e-3, "max_abs_err": max_abs, "batch": 4, "dim": 64}
