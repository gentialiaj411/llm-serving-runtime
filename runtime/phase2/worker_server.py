from __future__ import annotations

import asyncio
import collections
from contextlib import asynccontextmanager
import json
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from runtime.phase2.kv_allocator import PagedKVAllocator

CANCELLED_TTL_S = float(os.getenv("WORKER_CANCELLED_TTL_S", "3600"))
CANCELLED_CACHE_MAX = int(os.getenv("WORKER_CANCELLED_CACHE_MAX", "10000"))


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
        torch_dtype_name = os.getenv("HF_TORCH_DTYPE", "float32")
        device = os.getenv("HF_DEVICE", "cpu")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, torch_dtype_name)
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.bytes_per_elem = torch.empty((), dtype=dtype).element_size()
        self.bytes_per_token = self._bytes_per_token()

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

    def next_token(self, state: ActiveState) -> str:
        assert state.input_ids is not None
        with self.torch.no_grad():
            if not state.prompt_prefilled:
                outputs = self.model(
                    input_ids=state.input_ids,
                    attention_mask=state.attention_mask,
                    use_cache=True,
                )
                state.prompt_prefilled = True
            else:
                assert state.last_token_id is not None
                input_ids = self.torch.tensor([[state.last_token_id]], device=self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                )

        state.past_key_values = outputs.past_key_values
        next_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        state.last_token_id = next_id
        return self.tokenizer.decode([next_id], skip_special_tokens=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _allocator, _batch_task
    total_blocks = int(os.getenv("KV_TOTAL_BLOCKS", "4096"))
    block_size_tokens = int(os.getenv("KV_BLOCK_SIZE_TOKENS", "16"))
    _allocator = PagedKVAllocator(total_blocks=total_blocks, block_size_tokens=block_size_tokens)
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
            backend = _get_backend()
            assert _allocator is not None
            alloc = _allocator.allocate_for_tokens(
                pending.req.request_id,
                pending.req.max_tokens,
            )
            if alloc is None:
                # No KV capacity right now; request waits for next scheduling cycle.
                _waiting.appendleft(pending)
                break

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

            backend = _get_backend()
            token = (
                backend.next_token(state)
                if backend is not None
                else state.words[state.cursor % len(state.words)]
            )
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
                _allocator.free_request(rid)
            _cancelled.pop(rid, None)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    await _queue.put(Pending(req=req, fut=fut))
    return await fut


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
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
    return base
