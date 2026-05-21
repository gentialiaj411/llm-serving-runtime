from __future__ import annotations

import asyncio
import collections
from contextlib import asynccontextmanager
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from runtime.phase2.kv_allocator import PagedKVAllocator

class _JsonFormatter(logging.Formatter):
    _SKIP = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        out: dict[str, Any] = {
            "ts_unix_ms": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k not in self._SKIP:
                out[k] = v
        return json.dumps(out, default=str)


def _configure_logging() -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        root.addHandler(handler)
    root.setLevel(level)


_log = logging.getLogger("worker")

CANCELLED_TTL_S = float(os.getenv("WORKER_CANCELLED_TTL_S", "3600"))
CANCELLED_CACHE_MAX = int(os.getenv("WORKER_CANCELLED_CACHE_MAX", "10000"))
MAX_KV_RETRIES = int(os.getenv("WORKER_MAX_KV_RETRIES", "100"))
ADMISSION_TIMEOUT_S = float(os.getenv("WORKER_ADMISSION_TIMEOUT_S", "30.0"))
KV_BACKPRESSURE_PCT = float(os.getenv("WORKER_KV_BACKPRESSURE_PCT", "90.0"))


class GenerateRequest(BaseModel):
    request_id: str
    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = 0.0
    deadline_unix_ms: int | None = None


@dataclass
class Pending:
    req: GenerateRequest
    fut: asyncio.Future
    stream_queue: asyncio.Queue | None = None
    admitted_at: float = field(default_factory=time.time)
    kv_retry_count: int = 0


@dataclass
class ActiveState:
    req: GenerateRequest
    fut: asyncio.Future
    stream_queue: asyncio.Queue | None
    words: list[str]
    generated: list[str]
    cursor: int
    input_ids: Any | None = None
    attention_mask: Any | None = None
    past_key_values: Any | None = None
    last_token_id: int | None = None
    prompt_prefilled: bool = False
    next_logits: Any | None = None
    generated_token_ids: list[int] | None = None


_queue: asyncio.Queue[Pending] = asyncio.Queue()
_cancelled: collections.OrderedDict[str, float] = collections.OrderedDict()
_active: dict[str, ActiveState] = {}
_waiting: collections.deque[Pending] = collections.deque()
_allocator: PagedKVAllocator | None = None
_batch_task: asyncio.Task | None = None
_backend: Any | None = None
_backend_name = os.getenv("PHASE2_BACKEND", "synthetic").strip().lower()


class TransformersBackend:
    def __init__(self) -> None:
        model_id = os.getenv("HF_MODEL_ID", "sshleifer/tiny-gpt2")
        draft_model_id = os.getenv("HF_DRAFT_MODEL_ID", "sshleifer/tiny-gpt2")
        torch_dtype_name = os.getenv("HF_TORCH_DTYPE", "float32")
        device = os.getenv("HF_DEVICE", "cpu")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.speculative = os.getenv("PHASE2_SPECULATIVE", "0") == "1"
        self.spec_k = max(1, int(os.getenv("PHASE2_SPEC_K", "4")))
        dtype = getattr(torch, torch_dtype_name)
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.speculative:
            draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id)
            if self.tokenizer.get_vocab() != draft_tokenizer.get_vocab():
                model_id = os.getenv("HF_SPEC_TARGET_MODEL_ID", "EleutherAI/pythia-410m")
                draft_model_id = os.getenv("HF_SPEC_DRAFT_MODEL_ID", "EleutherAI/pythia-70m")
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id)
                if self.tokenizer.get_vocab() != draft_tokenizer.get_vocab():
                    raise RuntimeError("speculative decoding requires compatible draft and target tokenizers")
            self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_id, torch_dtype=dtype)
            self.draft_model.to(device)
            self.draft_model.eval()
        else:
            self.draft_model = None
        self.model_id = model_id
        self.draft_model_id = draft_model_id if self.speculative else None
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.bytes_per_elem = torch.empty((), dtype=dtype).element_size()
        self.bytes_per_token = self._bytes_per_token()
        self.spec_proposed = 0
        self.spec_accepted = 0

    def _bytes_per_token(self) -> int:
        config = self.model.config
        n_layer = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 1)))
        n_head = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 1)))
        hidden_size = int(getattr(config, "hidden_size", getattr(config, "n_embd", n_head)))
        head_dim = hidden_size // max(1, n_head)
        return max(1, n_layer * n_head * head_dim * 2 * self.bytes_per_elem)

    def init_state(self, state: ActiveState) -> None:
        encoded = self.tokenizer(state.req.prompt or "hello", return_tensors="pt")
        state.input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        state.attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
        state.generated_token_ids = []

    def _prefill_target(self, state: ActiveState) -> None:
        assert state.input_ids is not None
        if state.prompt_prefilled:
            return
        with self.torch.no_grad():
            outputs = self.model(
                input_ids=state.input_ids,
                attention_mask=state.attention_mask,
                use_cache=True,
            )
        state.past_key_values = self._normalize_past_key_values(outputs.past_key_values)
        state.next_logits = outputs.logits[:, -1, :]
        state.prompt_prefilled = True

    def _past_length(self, state: ActiveState) -> int:
        assert state.input_ids is not None
        generated_count = len(state.generated_token_ids or [])
        return int(state.input_ids.shape[1]) + generated_count

    def _cache_length(self, state: ActiveState) -> int:
        if state.past_key_values is None:
            return 0
        if hasattr(state.past_key_values, "get_seq_length"):
            return int(state.past_key_values.get_seq_length())
        first_layer = state.past_key_values[0]
        first_tensor = next(tensor for tensor in first_layer if tensor is not None)
        return int(first_tensor.shape[-2])

    def _normalize_past_key_values(self, past_key_values: Any) -> Any:
        if past_key_values is None or hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values
        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            return past_key_values
        try:
            return DynamicCache(past_key_values, config=self.model.config)
        except TypeError:
            return DynamicCache(past_key_values)

    def _concat_past_key_values(self, states: list[ActiveState]) -> Any:
        first = states[0].past_key_values
        if first is None:
            return None
        if hasattr(first, "to_legacy_cache"):
            first = first.to_legacy_cache()
            legacy = [
                state.past_key_values.to_legacy_cache()
                if hasattr(state.past_key_values, "to_legacy_cache")
                else state.past_key_values
                for state in states
            ]
        else:
            legacy = [state.past_key_values for state in states]
        combined = tuple(
            tuple(
                None
                if layers[0][layer_idx] is None
                else self.torch.cat([layer[layer_idx] for layer in layers], dim=0)
                for layer_idx in range(len(layers[0]))
            )
            for layers in zip(*legacy)
        )
        return self._normalize_past_key_values(combined)

    def _split_past_key_values(self, past_key_values: Any, count: int) -> list[Any]:
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        split: list[list[tuple[Any, ...]]] = [[] for _ in range(count)]
        for layer in past_key_values:
            layer_chunks = [None if tensor is None else tensor.split(1, dim=0) for tensor in layer]
            for idx in range(count):
                split[idx].append(tuple(None if chunks is None else chunks[idx] for chunks in layer_chunks))
        per_state = [tuple(layers) for layers in split]
        return [self._normalize_past_key_values(state_cache) for state_cache in per_state]

    def next_token_batch(self, states: list[ActiveState]) -> dict[str, str]:
        if self.speculative:
            return {state.req.request_id: self.next_token(state) for state in states}

        emitted: dict[str, str] = {}

        prefill_groups: dict[int, list[ActiveState]] = collections.defaultdict(list)
        decode_groups: dict[int, list[ActiveState]] = collections.defaultdict(list)
        for state in states:
            assert state.input_ids is not None
            if state.prompt_prefilled:
                decode_groups[self._cache_length(state)].append(state)
            else:
                prefill_groups[int(state.input_ids.shape[1])].append(state)

        with self.torch.no_grad():
            for group in prefill_groups.values():
                input_ids = self.torch.cat([state.input_ids for state in group], dim=0)
                attention_mask = None
                if all(state.attention_mask is not None for state in group):
                    attention_mask = self.torch.cat([state.attention_mask for state in group], dim=0)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                past_values = self._split_past_key_values(outputs.past_key_values, len(group))
                logits = outputs.logits[:, -1, :]
                for idx, state in enumerate(group):
                    state.past_key_values = past_values[idx]
                    state.prompt_prefilled = True
                    next_id = int(logits[idx].argmax(dim=-1).item())
                    state.last_token_id = next_id
                    assert state.generated_token_ids is not None
                    state.generated_token_ids.append(next_id)
                    emitted[state.req.request_id] = self.tokenizer.decode([next_id], skip_special_tokens=True)

            for group in decode_groups.values():
                input_ids = self.torch.tensor([[state.last_token_id] for state in group], device=self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=self._concat_past_key_values(group),
                    use_cache=True,
                )
                past_values = self._split_past_key_values(outputs.past_key_values, len(group))
                logits = outputs.logits[:, -1, :]
                for idx, state in enumerate(group):
                    next_id = int(logits[idx].argmax(dim=-1).item())
                    state.past_key_values = past_values[idx]
                    state.last_token_id = next_id
                    assert state.generated_token_ids is not None
                    state.generated_token_ids.append(next_id)
                    emitted[state.req.request_id] = self.tokenizer.decode([next_id], skip_special_tokens=True)

        return emitted

    def next_tokens_batch(self, states: list[ActiveState], max_tokens: int) -> dict[str, list[str]]:
        if self.speculative or max_tokens <= 1:
            return {rid: [token] for rid, token in self.next_token_batch(states).items()}
        if not states:
            return {}

        prefixes = []
        lengths = []
        for state in states:
            assert state.input_ids is not None
            assert state.generated_token_ids is not None
            prefix = state.input_ids
            if state.generated_token_ids:
                generated = self.torch.tensor([state.generated_token_ids], device=self.device)
                prefix = self.torch.cat([prefix, generated], dim=1)
            prefixes.append(prefix)
            lengths.append(int(prefix.shape[1]))

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        max_len = max(lengths)
        padded = []
        masks = []
        for prefix, length in zip(prefixes, lengths):
            pad = max_len - length
            if pad:
                pad_tensor = self.torch.full((1, pad), int(pad_id), dtype=prefix.dtype, device=self.device)
                prefix = self.torch.cat([prefix, pad_tensor], dim=1)
            padded.append(prefix)
            masks.append(
                self.torch.cat(
                    [
                        self.torch.ones((1, length), dtype=self.torch.long, device=self.device),
                        self.torch.zeros((1, pad), dtype=self.torch.long, device=self.device),
                    ],
                    dim=1,
                )
            )

        with self.torch.no_grad():
            input_ids = self.torch.cat(padded, dim=0)
            attention_mask = self.torch.cat(masks, dim=0)
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=int(pad_id),
            )

        result: dict[str, list[str]] = {}
        for idx, state in enumerate(states):
            token_ids = [int(token_id) for token_id in outputs[idx, max_len : max_len + max_tokens].tolist()]
            state.prompt_prefilled = False
            state.past_key_values = None
            state.next_logits = None
            state.generated_token_ids.extend(token_ids)
            state.last_token_id = token_ids[-1] if token_ids else state.last_token_id
            result[state.req.request_id] = [
                self.tokenizer.decode([token_id], skip_special_tokens=True) for token_id in token_ids
            ]
        return result

    def next_token(self, state: ActiveState) -> str:
        self._prefill_target(state)
        assert state.next_logits is not None
        next_id = int(state.next_logits.argmax(dim=-1).item())
        input_ids = self.torch.tensor([[next_id]], device=self.device)
        with self.torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=state.past_key_values,
                use_cache=True,
            )
        state.past_key_values = self._normalize_past_key_values(outputs.past_key_values)
        state.next_logits = outputs.logits[:, -1, :]
        state.last_token_id = next_id
        assert state.generated_token_ids is not None
        state.generated_token_ids.append(next_id)
        return self.tokenizer.decode([next_id], skip_special_tokens=True)

    def next_tokens(self, state: ActiveState, max_tokens: int) -> list[str]:
        if not self.speculative:
            return [self.next_token(state)]
        assert self.draft_model is not None
        assert state.input_ids is not None
        assert state.generated_token_ids is not None
        self._prefill_target(state)
        k = min(self.spec_k, max_tokens)
        prefix = state.input_ids
        if state.generated_token_ids:
            generated = self.torch.tensor([state.generated_token_ids], device=self.device)
            prefix = self.torch.cat([prefix, generated], dim=1)

        proposed: list[int] = []
        with self.torch.no_grad():
            draft_outputs = self.draft_model(input_ids=prefix, use_cache=True)
            draft_past = draft_outputs.past_key_values
            draft_logits = draft_outputs.logits[:, -1, :]
            for _ in range(k):
                draft_id = int(draft_logits.argmax(dim=-1).item())
                proposed.append(draft_id)
                draft_input = self.torch.tensor([[draft_id]], device=self.device)
                draft_outputs = self.draft_model(
                    input_ids=draft_input,
                    past_key_values=draft_past,
                    use_cache=True,
                )
                draft_past = draft_outputs.past_key_values
                draft_logits = draft_outputs.logits[:, -1, :]

            old_past = state.past_key_values
            proposed_tensor = self.torch.tensor([proposed], device=self.device)
            target_outputs = self.model(
                input_ids=proposed_tensor,
                past_key_values=old_past,
                use_cache=True,
            )

        emitted: list[int] = []
        rejected = False
        assert state.next_logits is not None
        for i, draft_id in enumerate(proposed):
            logits = state.next_logits if i == 0 else target_outputs.logits[:, i - 1, :]
            target_id = int(logits.argmax(dim=-1).item())
            self.spec_proposed += 1
            if draft_id == target_id:
                emitted.append(draft_id)
                self.spec_accepted += 1
            else:
                emitted.append(target_id)
                rejected = True
                break

        emitted_tensor = self.torch.tensor([emitted], device=self.device)
        if rejected:
            with self.torch.no_grad():
                refresh = self.model(input_ids=emitted_tensor, past_key_values=old_past, use_cache=True)
            state.past_key_values = self._normalize_past_key_values(refresh.past_key_values)
            state.next_logits = refresh.logits[:, -1, :]
        else:
            state.past_key_values = self._normalize_past_key_values(target_outputs.past_key_values)
            state.next_logits = target_outputs.logits[:, len(emitted) - 1, :]

        state.generated_token_ids.extend(emitted)
        state.last_token_id = emitted[-1]
        return [self.tokenizer.decode([token_id], skip_special_tokens=True) for token_id in emitted]

    def metrics(self) -> dict[str, Any]:
        acceptance_rate = (self.spec_accepted / self.spec_proposed) if self.spec_proposed else 0.0
        return {
            "phase2_backend": "transformers",
            "model_id": self.model_id,
            "speculative_enabled": self.speculative,
            "draft_model_id": self.draft_model_id,
            "speculative_k": self.spec_k if self.speculative else 0,
            "speculative_proposed_tokens": self.spec_proposed,
            "speculative_accepted_tokens": self.spec_accepted,
            "speculative_acceptance_rate": acceptance_rate,
        }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _allocator, _batch_task
    _configure_logging()
    total_blocks = int(os.getenv("KV_TOTAL_BLOCKS", "4096"))
    block_size_tokens = int(os.getenv("KV_BLOCK_SIZE_TOKENS", "16"))
    _allocator = PagedKVAllocator(total_blocks=total_blocks, block_size_tokens=block_size_tokens)
    _log.info("worker starting", extra={"total_blocks": total_blocks, "block_size_tokens": block_size_tokens})
    _batch_task = asyncio.create_task(_continuous_batch_loop())
    try:
        yield
    finally:
        if _batch_task is not None:
            _batch_task.cancel()
            try:
                await _batch_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="phase2-worker", lifespan=lifespan)


def _get_backend() -> Any | None:
    global _allocator, _backend
    if _backend_name == "synthetic":
        return None
    if _backend_name != "transformers":
        raise RuntimeError("PHASE2_BACKEND must be one of: synthetic, transformers")
    if _backend is None:
        _backend = TransformersBackend()
        total_blocks = int(os.getenv("KV_TOTAL_BLOCKS", "4096"))
        block_size_tokens = int(os.getenv("KV_BLOCK_SIZE_TOKENS", "16"))
        _allocator = PagedKVAllocator(
            total_blocks=total_blocks,
            block_size_tokens=block_size_tokens,
            bytes_per_token=int(_backend.bytes_per_token),
        )
    return _backend


def _prune_cancelled(now: float | None = None) -> None:
    now = now or time.time()
    expired = [rid for rid, ts in _cancelled.items() if now - ts > CANCELLED_TTL_S]
    for rid in expired:
        _cancelled.pop(rid, None)
    while len(_cancelled) > CANCELLED_CACHE_MAX:
        _cancelled.popitem(last=False)


def _mark_cancelled(request_id: str) -> None:
    _cancelled[request_id] = time.time()
    _cancelled.move_to_end(request_id)
    _prune_cancelled()


def _is_cancelled(request_id: str) -> bool:
    _prune_cancelled()
    return request_id in _cancelled


async def _continuous_batch_loop() -> None:
    # Orca-style idea: schedule at iteration boundaries, admitting new requests continuously.
    max_active = 32
    decode_step_ms = 2
    while True:
        # Pull at least one request if system is idle.
        if not _waiting and not _active:
            _waiting.append(await _queue.get())

        # Non-blocking drain of new arrivals.
        while True:
            try:
                _waiting.append(_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Admit waiting requests into active decode set.
        while _waiting and len(_active) < max_active:
            pending = _waiting.popleft()

            # Fail requests that have waited past the admission deadline.
            wait_s = time.time() - pending.admitted_at
            if wait_s > ADMISSION_TIMEOUT_S:
                _log.warning(
                    "request timed out in admission queue",
                    extra={"request_id": pending.req.request_id, "wait_s": round(wait_s, 2)},
                )
                if not pending.fut.done():
                    pending.fut.set_result({"request_id": pending.req.request_id, "text": "", "timed_out": True})
                if pending.stream_queue is not None:
                    pending.stream_queue.put_nowait({"type": "timed_out"})
                continue

            # Fail requests that have exhausted their KV retry budget.
            if pending.kv_retry_count >= MAX_KV_RETRIES:
                _log.error(
                    "KV allocation retry limit exceeded; dropping request",
                    extra={"request_id": pending.req.request_id, "retries": pending.kv_retry_count},
                )
                if not pending.fut.done():
                    pending.fut.set_result({"request_id": pending.req.request_id, "text": "", "timed_out": True})
                if pending.stream_queue is not None:
                    pending.stream_queue.put_nowait({"type": "timed_out"})
                continue

            backend = _get_backend()
            assert _allocator is not None
            alloc = _allocator.allocate_for_tokens(
                pending.req.request_id,
                pending.req.max_tokens,
            )
            if alloc is None:
                # No KV capacity right now; request waits for next scheduling cycle.
                pending.kv_retry_count += 1
                if pending.kv_retry_count == 1:
                    _log.warning("KV memory exhausted; request queued for retry", extra={"request_id": pending.req.request_id})
                _waiting.appendleft(pending)
                break

            _log.info("request admitted to decode set", extra={"request_id": pending.req.request_id})
            words = pending.req.prompt.split() or ["hello"]
            state = ActiveState(
                req=pending.req,
                fut=pending.fut,
                stream_queue=pending.stream_queue,
                words=words,
                generated=[],
                cursor=0,
            )
            if backend is not None:
                backend.init_state(state)
            _active[pending.req.request_id] = state

        if not _active:
            await asyncio.sleep(0.001)
            continue

        # One decode iteration: advance every active request by one token.
        await asyncio.sleep(decode_step_ms / 1000.0)
        now_ms = int(time.time() * 1000)
        finished_ids: list[str] = []
        ready: list[ActiveState] = []

        for rid, state in list(_active.items()):
            if _is_cancelled(rid):
                if not state.fut.done():
                    state.fut.set_result({"request_id": rid, "text": "", "cancelled": True})
                if state.stream_queue is not None:
                    state.stream_queue.put_nowait({"type": "cancelled"})
                finished_ids.append(rid)
                continue

            if state.req.deadline_unix_ms is not None and now_ms > state.req.deadline_unix_ms:
                if not state.fut.done():
                    state.fut.set_result({"request_id": rid, "text": "", "timed_out": True})
                if state.stream_queue is not None:
                    state.stream_queue.put_nowait({"type": "timed_out"})
                finished_ids.append(rid)
                continue

            ready.append(state)

        backend = _get_backend()
        batch_tokens: dict[str, str] = {}
        batched_ids: set[str] = set()
        if backend is not None and ready and not getattr(backend, "speculative", False):
            batch_steps = max(1, int(os.getenv("PHASE2_BATCH_DECODE_STEPS", "16")))
            batch_steps = min(batch_steps, *(state.req.max_tokens - len(state.generated) for state in ready))
            batch_token_lists = backend.next_tokens_batch(ready, batch_steps)
            for state in ready:
                rid = state.req.request_id
                batched_ids.add(rid)
                for token in batch_token_lists[rid]:
                    state.generated.append(token)
                    state.cursor += 1
                    if state.stream_queue is not None:
                        state.stream_queue.put_nowait({"type": "token", "token": token, "text": token, "index": state.cursor - 1})
                if len(state.generated) >= state.req.max_tokens:
                    text = " ".join(state.generated)
                    if not state.fut.done():
                        state.fut.set_result({"request_id": rid, "text": text, "cancelled": False})
                    if state.stream_queue is not None:
                        state.stream_queue.put_nowait({"type": "done", "text": text})
                    finished_ids.append(rid)

        for state in ready:
            rid = state.req.request_id
            if rid in finished_ids or rid in batched_ids:
                continue
            backend = _get_backend()
            remaining = state.req.max_tokens - len(state.generated)
            tokens = (
                [batch_tokens[rid]]
                if rid in batch_tokens
                else backend.next_tokens(state, remaining)
                if backend is not None
                else [state.words[state.cursor % len(state.words)]]
            )
            for token in tokens[:remaining]:
                state.generated.append(token)
                state.cursor += 1
                if state.stream_queue is not None:
                    state.stream_queue.put_nowait({"type": "token", "token": token, "text": token, "index": state.cursor - 1})

            if len(state.generated) >= state.req.max_tokens:
                text = " ".join(state.generated)
                if not state.fut.done():
                    state.fut.set_result({"request_id": rid, "text": text, "cancelled": False})
                if state.stream_queue is not None:
                    state.stream_queue.put_nowait({"type": "done", "text": text})
                finished_ids.append(rid)

        for rid in finished_ids:
            _active.pop(rid, None)
            if _allocator is not None:
                try:
                    _allocator.free_request(rid)
                except Exception:
                    _log.error("failed to free KV allocation", extra={"request_id": rid}, exc_info=True)
            _cancelled.pop(rid, None)


def _check_backpressure() -> None:
    if _allocator is not None:
        stats = _allocator.stats()
        if stats["used_pct"] >= KV_BACKPRESSURE_PCT:
            _log.warning("KV backpressure triggered; rejecting new request", extra={"used_pct": stats["used_pct"]})
            raise HTTPException(
                status_code=503,
                detail=f"KV memory at {stats['used_pct']:.1f}% capacity; try again later",
            )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    _check_backpressure()
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    await _queue.put(Pending(req=req, fut=fut))
    return await fut


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    _check_backpressure()
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    stream_queue: asyncio.Queue = asyncio.Queue()
    await _queue.put(Pending(req=req, fut=fut, stream_queue=stream_queue))

    async def events() -> Any:
        while True:
            event = await stream_queue.get()
            yield json.dumps(event) + "\n"
            if event.get("type") in {"done", "cancelled", "timed_out"}:
                break
        if not fut.done():
            fut.cancel()

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/cancel/{request_id}")
async def cancel(request_id: str) -> dict[str, str]:
    _mark_cancelled(request_id)
    return {"request_id": request_id, "status": "cancel_accepted"}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    base: dict[str, Any] = {
        "queue_waiting": len(_waiting),
        "active_decode": len(_active),
    }
    if _allocator is not None:
        base.update(_allocator.stats())
    if _backend is not None:
        base.update(_backend.metrics())
    return base
