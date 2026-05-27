from __future__ import annotations

import os


def test_int4_quantized_generation_non_empty() -> None:
    import pytest

    pytest.importorskip("awq")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    os.environ["PHASE2_BACKEND"] = "transformers"
    os.environ["PHASE2_QUANT"] = "int4"
    os.environ["HF_AWQ_MODEL_ID"] = os.getenv("HF_AWQ_MODEL_ID", "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ")
    os.environ["HF_TORCH_DTYPE"] = "float16"
    os.environ["HF_DEVICE"] = "cuda"

    from runtime.phase2.worker_server import ActiveState, GenerateRequest, TransformersBackend

    backend = TransformersBackend()
    state = ActiveState(
        req=GenerateRequest(request_id="q4", prompt="hello world", max_tokens=8),
        fut=None,  # type: ignore[arg-type]
        stream_queue=None,
        words=[],
        generated=[],
        cursor=0,
    )
    backend.init_state(state)
    tokens = backend.next_tokens(state, 8)
    assert "".join(tokens).strip()
