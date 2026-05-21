from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field


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


_configure_logging()
_log = logging.getLogger("phase1")

app = FastAPI(title="phase1-openai-shim")
_transformers_backend: dict[str, Any] = {}
_transformers_lock = threading.Lock()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = 0.0
    stream: bool = False


def _deterministic_generate(prompt: str, max_tokens: int) -> str:
    words = prompt.split()
    if not words:
        words = ["hello"]
    out = []
    for i in range(max_tokens):
        out.append(words[i % len(words)])
    return " ".join(out)


def _backend_mode() -> str:
    return os.getenv("PHASE1_BACKEND", "stub").strip().lower()


def _load_transformers(model_id: str) -> dict[str, Any]:
    with _transformers_lock:
        loaded_model = _transformers_backend.get("model_id")
        if loaded_model == model_id:
            return _transformers_backend

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="PHASE1_BACKEND=transformers requires torch and transformers to be installed",
            ) from exc

        device = os.getenv("PHASE1_TRANSFORMERS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        dtype_name = os.getenv("PHASE1_TRANSFORMERS_DTYPE", "float16" if device.startswith("cuda") else "float32")
        dtype = getattr(torch, dtype_name)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model.to(device)
        model.eval()

        _transformers_backend.clear()
        _transformers_backend.update({
            "model_id": model_id,
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
        })
        return _transformers_backend


def _transformers_prompt(req: ChatRequest) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in req.messages) + "\nassistant:"


def _transformers_generate(req: ChatRequest) -> tuple[str, dict[str, int]]:
    backend = _load_transformers(req.model)
    tokenizer = backend["tokenizer"]
    model = backend["model"]
    device = backend["device"]
    prompt = _transformers_prompt(req)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0,
        temperature=max(req.temperature, 1e-5),
        pad_token_id=tokenizer.eos_token_id,
    )
    completion_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    usage = {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "completion_tokens": int(completion_ids.shape[-1]),
        "total_tokens": int(output_ids.shape[-1]),
    }
    return text, usage


def _stream_chunk(model: str, content: str) -> str:
    payload = {
        "id": "chatcmpl-phase1",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_completion(req: ChatRequest, prompt: str) -> Any:
    if _backend_mode() == "transformers":
        async for chunk in _stream_transformers_completion(req):
            yield chunk
        return

    await asyncio.sleep(float(os.getenv("PHASE1_SIMULATED_PREFILL_MS", "20")) / 1000.0)
    decode_ms = float(os.getenv("PHASE1_SIMULATED_DECODE_MS", "2"))
    words = prompt.split() or ["hello"]
    for i in range(req.max_tokens):
        token = words[i % len(words)]
        content = token if i == 0 else f" {token}"
        yield _stream_chunk(req.model, content)
        await asyncio.sleep(decode_ms / 1000.0)

    done = {
        "id": "chatcmpl-phase1",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": max(1, len(prompt.split())),
            "completion_tokens": req.max_tokens,
            "total_tokens": max(1, len(prompt.split())) + req.max_tokens,
        },
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


def _next_streamer_text(streamer: Any) -> tuple[bool, str]:
    try:
        return True, next(streamer)
    except StopIteration:
        return False, ""


async def _stream_transformers_completion(req: ChatRequest) -> Any:
    try:
        from transformers import TextIteratorStreamer
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="PHASE1_BACKEND=transformers requires transformers to be installed",
        ) from exc

    backend = _load_transformers(req.model)
    tokenizer = backend["tokenizer"]
    model = backend["model"]
    device = backend["device"]
    prompt = _transformers_prompt(req)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": req.max_tokens,
        "do_sample": req.temperature > 0,
        "temperature": max(req.temperature, 1e-5),
        "pad_token_id": tokenizer.eos_token_id,
    }
    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs, daemon=True)
    thread.start()

    completion_text = ""
    while True:
        ok, text = await asyncio.to_thread(_next_streamer_text, streamer)
        if not ok:
            break
        if text:
            completion_text += text
            yield _stream_chunk(req.model, text)

    completion_tokens = len(tokenizer(completion_text).input_ids) if completion_text else 0
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    done = {
        "id": "chatcmpl-phase1",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(req: ChatRequest) -> dict[str, Any] | Response:
    t0 = time.perf_counter()
    prompt = "\n".join([m.content for m in req.messages])
    _log.info("chat completion request", extra={"model": req.model, "stream": req.stream, "max_tokens": req.max_tokens})

    if req.stream:
        return StreamingResponse(_stream_completion(req, prompt), media_type="text/event-stream")

    # Phase 1 baseline: request-by-request execution, no batching.
    if _backend_mode() == "transformers":
        text, usage = await asyncio.to_thread(_transformers_generate, req)
    else:
        await asyncio.sleep(float(os.getenv("PHASE1_SIMULATED_PREFILL_MS", "20")) / 1000.0)
        text = _deterministic_generate(prompt, req.max_tokens)
        usage = {
            "prompt_tokens": max(1, len(prompt.split())),
            "completion_tokens": max(1, len(text.split())),
            "total_tokens": max(1, len(prompt.split())) + max(1, len(text.split())),
        }
    latency_ms = (time.perf_counter() - t0) * 1000.0
    _log.info("chat completion done", extra={"model": req.model, "latency_ms": round(latency_ms, 2)})

    return {
        "id": "chatcmpl-phase1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "_debug": {"latency_ms": latency_ms},
    }
