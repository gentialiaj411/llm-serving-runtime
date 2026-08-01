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
from runtime.phase2.paged_kv_cache import BlockPagedBatchCache, BlockPagedCache
from runtime.phase2.paged_kv_kernel import GpuKVBlockPool
from runtime.phase2.lora_manager import LoRAManager, load_adapter_config_from_env
from runtime.phase2.prefix_cache import PrefixBlockCache

CANCELLED_TTL_S = float(os.getenv("WORKER_CANCELLED_TTL_S", "3600"))
CANCELLED_CACHE_MAX = int(os.getenv("WORKER_CANCELLED_CACHE_MAX", "10000"))


class GenerateRequest(BaseModel):
    request_id: str
    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = 0.0
    deadline_unix_ms: int | None = None
    prefill_handoff_id: str | None = None
    adapter: str = "base"


class PrefillRequest(BaseModel):
    request_id: str
    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = 0.0
    deadline_unix_ms: int | None = None


class DecodeRequest(BaseModel):
    request_id: str
    prompt: str = ""
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = 0.0
    deadline_unix_ms: int | None = None
    prefill_handoff_id: str | None = None


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
    token_capacity: int = 0
    contiguous_kv_hold: Any | None = None
    last_token_id: int | None = None
    prompt_prefilled: bool = False
    next_logits: Any | None = None
    generated_token_ids: list[int] | None = None
    prefix_entry_id: str | None = None
    should_insert_prefix: bool = False


_queue: asyncio.Queue[Pending] = asyncio.Queue()
_prefix_cache: PrefixBlockCache | None = None
_cancelled: collections.OrderedDict[str, float] = collections.OrderedDict()
_active: dict[str, ActiveState] = {}
_waiting: collections.deque[Pending] = collections.deque()
_allocator: PagedKVAllocator | None = None
_batch_task: asyncio.Task | None = None
_backend: Any | None = None
_backend_name = os.getenv("PHASE2_BACKEND", "synthetic").strip().lower()
_prefill_handoffs: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
_prefill_handoff_ttl_s = float(os.getenv("WORKER_PREFILL_HANDOFF_TTL_S", "300"))
_prefill_handoff_max = int(os.getenv("WORKER_PREFILL_HANDOFF_MAX", "10000"))
_terminal_cache: collections.OrderedDict[str, tuple[float, dict[str, Any]]] = collections.OrderedDict()
_terminal_cache_ttl_s = float(os.getenv("WORKER_TERMINAL_CACHE_TTL_S", "3600"))
_terminal_cache_max = int(os.getenv("WORKER_TERMINAL_CACHE_MAX", "10000"))
_peak_torch_cuda_bytes: int = 0


def _record_cuda_peak() -> None:
    global _peak_torch_cuda_bytes
    try:
        import torch

        if torch.cuda.is_available():
            _peak_torch_cuda_bytes = max(_peak_torch_cuda_bytes, int(torch.cuda.max_memory_allocated()))
    except Exception:
        pass


_scheduler_metrics: dict[str, int] = {
    "decode_iterations_total": 0,
    "continuous_batches_total": 0,
    "batched_requests_total": 0,
    "peak_active_requests": 0,
    "max_batch_size": 0,
    "stream_completed_total": 0,
    "stream_cancelled_total": 0,
    "stream_timed_out_total": 0,
    "request_errors_total": 0,
}
_queued_request_ids: set[str] = set()


class TransformersBackend:
    def __init__(self) -> None:
        quant_mode = os.getenv("PHASE2_QUANT", "none").strip().lower()
        awq_model_id = os.getenv("HF_AWQ_MODEL_ID", "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ")
        model_id = awq_model_id if quant_mode == "int4" else os.getenv("HF_MODEL_ID", "sshleifer/tiny-gpt2")
        draft_model_id = os.getenv("HF_DRAFT_MODEL_ID", "sshleifer/tiny-gpt2")
        torch_dtype_name = os.getenv("HF_TORCH_DTYPE", "float32")
        device = os.getenv("HF_DEVICE", "cpu")
        self.quant_mode = quant_mode

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

        class _LocalTokenizer:
            def __init__(self, vocab_size: int = 256) -> None:
                self.vocab_size = vocab_size
                self.pad_token_id = 0
                self.eos_token_id = 1

            def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, Any]:
                ids = [2 + (abs(hash(tok)) % max(1, self.vocab_size - 2)) for tok in (text.split() or ["hello"])]
                return {
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                }

            def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
                if not token_ids:
                    return ""
                token_id = int(token_ids[0])
                if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                    return ""
                return f"tok{token_id}"

            def get_vocab(self) -> dict[str, int]:
                return {f"tok{i}": i for i in range(self.vocab_size)}

        self.speculative = os.getenv("PHASE2_SPECULATIVE", "0") == "1"
        if self.speculative and self.quant_mode == "int4":
            raise RuntimeError("PHASE2_SPECULATIVE=1 is not supported with PHASE2_QUANT=int4")
        self.spec_k = max(1, int(os.getenv("PHASE2_SPEC_K", "4")))
        dtype = getattr(torch, torch_dtype_name)
        self.torch = torch
        self.device = device
        local_random = model_id.strip().lower() == "local-random-gpt2"
        if local_random:
            config = GPT2Config(vocab_size=256, n_layer=2, n_head=2, n_embd=64)
            self.model = GPT2LMHeadModel(config)
            self.model.to(device)
            self.model.eval()
            self.tokenizer = _LocalTokenizer(vocab_size=256)
        else:
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
        if not local_random:
            if self.quant_mode == "int4":
                try:
                    from awq import AutoAWQForCausalLM
                except Exception as exc:
                    raise RuntimeError(f"AWQ runtime unavailable for PHASE2_QUANT=int4: {exc}") from exc
                self.model = AutoAWQForCausalLM.from_quantized(
                    model_id,
                    fuse_layers=False,
                    trust_remote_code=True,
                )
                self.model.to(device)
                self.model.eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
                self.model.to(device)
                self.model.eval()
        self.bytes_per_elem = torch.empty((), dtype=dtype).element_size()
        self.bytes_per_token = self._bytes_per_token()
        self.kv_backend = os.getenv("PHASE2_KV_BACKEND", "dynamic").strip().lower()
        self.kv_pool: GpuKVBlockPool | None = None
        if self.kv_backend == "paged":
            total_blocks = int(os.getenv("KV_TOTAL_BLOCKS", "4096"))
            block_size_tokens = int(os.getenv("KV_BLOCK_SIZE_TOKENS", "16"))
            config = self.model.config
            n_layer = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 1)))
            n_attn_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 1)))
            n_kv_heads = int(getattr(config, "num_key_value_heads", n_attn_heads))
            hidden_size = int(getattr(config, "hidden_size", getattr(config, "n_embd", n_attn_heads)))
            head_dim = hidden_size // max(1, n_attn_heads)
            self.kv_pool = GpuKVBlockPool(
                num_blocks=total_blocks,
                block_size_tokens=block_size_tokens,
                num_layers=n_layer,
                num_kv_heads=n_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
            self._enable_block_paged_attention()
            # Persistent continuous-batch slots (survive admission/completion).
            self._paged_persistent: BlockPagedBatchCache | None = None
            self._paged_persistent_capacity = 0
            self._cuda_graph_manager = None
            if os.getenv("PHASE2_CUDA_GRAPH", "0").strip() == "1" and self.torch.cuda.is_available():
                from runtime.phase2.cuda_graph_decode import CudaGraphDecodeManager

                self._cuda_graph_manager = CudaGraphDecodeManager(
                    model=self.model,
                    device=self.torch.device(self.device),
                    warmup_steps=int(os.getenv("PHASE2_CUDA_GRAPH_WARMUP", "3")),
                )
        else:
            self._paged_persistent = None
            self._paged_persistent_capacity = 0
            self._cuda_graph_manager = None
        self.lora_enabled = os.getenv("PHASE2_LORA", "0") == "1" and quant_mode != "int4" and not self.speculative
        self.lora_manager: LoRAManager | None = None
        if self.lora_enabled:
            adapter_names, adapter_paths = load_adapter_config_from_env()
            self.lora_manager = LoRAManager(
                self.model,
                enabled=True,
                adapter_names=adapter_names,
                adapter_paths=adapter_paths,
            )
            self.model = self.lora_manager.model
            self.lora_adapter_names = adapter_names
        else:
            self.lora_adapter_names = ["base"]

        self.spec_proposed = 0
        self.spec_accepted = 0
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()
            _record_cuda_peak()

    def _enable_block_paged_attention(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from runtime.phase2.block_paged_attention import block_paged_attention_forward

        ALL_ATTENTION_FUNCTIONS.register("block_paged", block_paged_attention_forward)
        self.model.config._attn_implementation = "block_paged"
        self._warmup_paged_attention_kernel()

    def _warmup_paged_attention_kernel(self) -> None:
        if self.kv_pool is None or self.kv_backend != "paged":
            return
        from runtime.phase2.paged_attention_triton import triton_available, warmup_paged_attention_kernel

        if not triton_available() or not self.torch.cuda.is_available():
            return
        config = self.model.config
        n_attn_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 1)))
        n_kv_heads = int(getattr(config, "num_key_value_heads", n_attn_heads))
        hidden_size = int(getattr(config, "hidden_size", getattr(config, "n_embd", n_attn_heads)))
        head_dim = hidden_size // max(1, n_attn_heads)
        num_kv_groups = max(1, n_attn_heads // max(1, n_kv_heads))
        warmup_paged_attention_kernel(
            self.kv_pool,
            num_heads=n_attn_heads,
            num_kv_groups=num_kv_groups,
            head_dim=head_dim,
            block_ids=[0],
        )

    def _bytes_per_token(self) -> int:
        config = self.model.config
        n_layer = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 1)))
        n_head = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 1)))
        hidden_size = int(getattr(config, "hidden_size", getattr(config, "n_embd", n_head)))
        head_dim = hidden_size // max(1, n_head)
        return max(1, n_layer * n_head * head_dim * 2 * self.bytes_per_elem)

    def _allocate_contiguous_hold(self, token_capacity: int) -> Any:
        config = self.model.config
        n_layer = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 1)))
        n_attn_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 1)))
        n_kv_heads = int(getattr(config, "num_key_value_heads", n_attn_heads))
        hidden_size = int(getattr(config, "hidden_size", getattr(config, "n_embd", n_attn_heads)))
        head_dim = hidden_size // max(1, n_attn_heads)
        seq = max(1, token_capacity)
        return self.torch.zeros(
            (n_layer, 2, n_kv_heads, seq, head_dim),
            dtype=self.model.dtype if hasattr(self.model, "dtype") else self.torch.float16,
            device=self.device,
        )

    def _create_kv_cache(self, token_capacity: int, block_ids: list[int]) -> Any | None:
        if self.kv_backend in {"static", "reserved"}:
            from transformers import StaticCache

            return StaticCache(config=self.model.config, max_cache_len=max(1, token_capacity))
        if self.kv_backend == "paged":
            if self.kv_pool is None:
                raise RuntimeError("paged KV backend requires GpuKVBlockPool")
            cache = BlockPagedCache(config=self.model.config, pool=self.kv_pool)
            cache.set_block_ids(block_ids)
            return cache
        return None

    def sync_kv_blocks(self, state: ActiveState, block_ids: list[int]) -> None:
        cache = state.past_key_values
        if isinstance(cache, BlockPagedCache):
            cache.sync_block_ids(block_ids)
        persistent = getattr(self, "_paged_persistent", None)
        if persistent is not None:
            slot = persistent.slot_of(state.req.request_id)
            if slot is not None:
                persistent.sync_block_ids_row(slot, list(block_ids))

    def release_paged_request(self, request_id: str) -> None:
        persistent = getattr(self, "_paged_persistent", None)
        if persistent is not None:
            persistent.release_slot(request_id)

    def _ensure_paged_persistent(self, min_batch: int, max_blocks: int) -> BlockPagedBatchCache:
        if self.kv_pool is None:
            raise RuntimeError("paged persistent batch requires GpuKVBlockPool")
        capacity = max(min_batch, int(os.getenv("PHASE2_MAX_ACTIVE", "8")), 1)
        capacity = max(capacity, min_batch)
        persistent = getattr(self, "_paged_persistent", None)
        if persistent is None or self._paged_persistent_capacity < capacity:
            # Grow: allocate a larger empty table; re-bind existing request rows.
            old = persistent
            persistent = BlockPagedBatchCache.empty(
                self.model.config,
                self.kv_pool,
                max_batch_size=capacity,
                max_blocks=max(max_blocks, 1),
            )
            if old is not None:
                for rid, old_slot in list(old._request_to_slot.items()):
                    new_slot = persistent.allocate_slot(rid)
                    persistent.sync_block_ids_row(new_slot, list(old.block_ids_rows[old_slot]))
                    persistent.seq_lens[new_slot] = old.seq_lens[old_slot]
            self._paged_persistent = persistent
            self._paged_persistent_capacity = capacity
        else:
            persistent.ensure_max_blocks(max_blocks)
        return persistent

    def occupied_tokens(self, state: ActiveState) -> int:
        if state.input_ids is not None:
            base = int(state.input_ids.shape[1])
        else:
            base = max(1, len(state.words))
        generated = len(state.generated_token_ids or [])
        if state.past_key_values is not None and hasattr(state.past_key_values, "get_seq_length"):
            return max(base + generated, int(state.past_key_values.get_seq_length()))
        return base + generated

    def init_state(self, state: ActiveState) -> None:
        encoded = None
        if state.input_ids is None:
            encoded = self.tokenizer(state.req.prompt or "hello", return_tensors="pt")
            state.input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask") if encoded is not None else state.attention_mask
        state.attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
        state.generated_token_ids = []

    def set_active_adapter(self, adapter: str | None) -> None:
        if self.lora_manager is not None:
            self.lora_manager.set_active(adapter or "base")

    def prompt_token_ids(self, state: ActiveState) -> list[int]:
        assert state.input_ids is not None
        return [int(x) for x in state.input_ids[0].tolist()]

    def clone_past_key_values(self, past_key_values: Any) -> Any:
        if past_key_values is None:
            return None
        from transformers.cache_utils import Cache, DynamicCache

        if isinstance(past_key_values, Cache):
            new_cache = DynamicCache(config=self.model.config)
            for layer_idx, layer in enumerate(past_key_values.layers):
                if not getattr(layer, "is_initialized", False):
                    continue
                keys = layer.keys
                values = layer.values
                if keys is None or values is None or keys.numel() == 0:
                    continue
                new_cache.update(keys.clone(), values.clone(), layer_idx)
            return new_cache

        if hasattr(past_key_values, "to_legacy_cache"):
            legacy = past_key_values.to_legacy_cache()
        else:
            legacy = past_key_values
        cloned = tuple(
            tuple(tensor.clone() if tensor is not None else None for tensor in layer)
            for layer in legacy
        )
        return self._normalize_past_key_values(cloned)

    def _prefill_target(self, state: ActiveState) -> None:
        assert state.input_ids is not None
        if state.prompt_prefilled:
            return
        with self.torch.no_grad():
            outputs = self.model(
                input_ids=state.input_ids,
                attention_mask=state.attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
            )
        state.past_key_values = self._normalize_past_key_values(outputs.past_key_values)
        state.next_logits = outputs.logits[:, -1, :]
        state.prompt_prefilled = True
        _try_insert_prefix_cache(state)

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
            from transformers.cache_utils import Cache, DynamicCache
        except Exception:
            return past_key_values
        if isinstance(past_key_values, Cache):
            return past_key_values
        try:
            return DynamicCache(past_key_values, config=self.model.config)
        except TypeError:
            return DynamicCache(past_key_values)

    def _concat_paged_caches(self, states: list[ActiveState]) -> BlockPagedBatchCache:
        if self.kv_pool is None:
            raise RuntimeError("paged concat requires GpuKVBlockPool")
        caches = [state.past_key_values for state in states]
        if not all(isinstance(c, BlockPagedCache) for c in caches):
            raise TypeError("paged batch decode requires BlockPagedCache per request")
        return BlockPagedBatchCache.from_caches(caches, self.kv_pool)

    def _split_paged_batch_cache(
        self, batch: BlockPagedBatchCache, states: list[ActiveState]
    ) -> None:
        for idx, state in enumerate(states):
            state.past_key_values = batch.extract_cache(idx)

    def _bind_states_to_persistent(self, states: list[ActiveState]) -> BlockPagedBatchCache:
        max_blocks = 1
        for state in states:
            cache = state.past_key_values
            if isinstance(cache, BlockPagedCache):
                max_blocks = max(max_blocks, len(cache.block_table_ids) or 1)
        persistent = self._ensure_paged_persistent(len(states), max_blocks)
        # Do not release slots for requests absent from this call — completions go through
        # release_paged_request so mid-batch just-prefilled rows stay bound.
        for state in states:
            rid = state.req.request_id
            if persistent.slot_of(rid) is None:
                if not isinstance(state.past_key_values, BlockPagedCache):
                    raise TypeError("binding persistent slot requires BlockPagedCache")
                persistent.bind_cache(rid, state.past_key_values)
            else:
                cache = state.past_key_values
                if isinstance(cache, BlockPagedCache):
                    slot = persistent.slot_of(rid)
                    assert slot is not None
                    persistent.sync_block_ids_row(slot, list(cache.block_table_ids))
                    persistent.seq_lens[slot] = int(cache.get_seq_length())
        return persistent

    def _paged_prefill_groups(self, states: list[ActiveState]) -> dict[str, str]:
        """Prefill unprefilled rows; emit first token from prefill logits."""
        emitted: dict[str, str] = {}
        if not states:
            return emitted

        prefill_groups: dict[int, list[ActiveState]] = collections.defaultdict(list)
        for state in states:
            assert state.input_ids is not None
            prefill_groups[int(state.input_ids.shape[1])].append(state)

        with self.torch.no_grad():
            for group in prefill_groups.values():
                input_ids = self.torch.cat([state.input_ids for state in group], dim=0)
                attention_mask = None
                if all(state.attention_mask is not None for state in group):
                    attention_mask = self.torch.cat([state.attention_mask for state in group], dim=0)
                if len(group) == 1:
                    past = group[0].past_key_values
                else:
                    past = self._concat_paged_caches(group)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                if isinstance(outputs.past_key_values, BlockPagedBatchCache):
                    batch_cache = outputs.past_key_values
                    past_values = [batch_cache.extract_cache(i) for i in range(len(group))]
                elif len(group) == 1:
                    past_values = [self._normalize_past_key_values(outputs.past_key_values)]
                else:
                    raise TypeError("expected BlockPagedBatchCache from batched paged prefill")
                logits = outputs.logits[:, -1, :]
                for idx, state in enumerate(group):
                    state.past_key_values = past_values[idx]
                    state.prompt_prefilled = True
                    next_id = int(logits.argmax(dim=-1).tolist()[idx])
                    state.last_token_id = next_id
                    assert state.generated_token_ids is not None
                    state.generated_token_ids.append(next_id)
                    emitted[state.req.request_id] = self.tokenizer.decode([next_id], skip_special_tokens=True)
        return emitted

    def _paged_decode_step(
        self,
        decode_states: list[ActiveState],
        batch_cache: BlockPagedBatchCache | None,
    ) -> tuple[dict[str, str], BlockPagedBatchCache | None]:
        """One ragged batched decode forward using persistent slot tables when possible."""
        emitted: dict[str, str] = {}
        input_ids = self.torch.tensor([[state.last_token_id] for state in decode_states], device=self.device)

        with self.torch.no_grad():
            persistent = self._bind_states_to_persistent(decode_states)
            batch_cache = persistent
            logits_row = None

            graph_mgr = getattr(self, "_cuda_graph_manager", None)
            if graph_mgr is not None and len(decode_states) >= 1:
                logits_row = graph_mgr.try_decode(
                    persistent=persistent,
                    decode_states=decode_states,
                    input_ids=input_ids,
                )

            if logits_row is None:
                if len(decode_states) == 1:
                    state = decode_states[0]
                    outputs = self.model(
                        input_ids=input_ids,
                        past_key_values=state.past_key_values,
                        use_cache=True,
                    )
                    new_cache = self._normalize_past_key_values(outputs.past_key_values)
                    state.past_key_values = new_cache
                    if isinstance(new_cache, BlockPagedCache):
                        persistent = self._bind_states_to_persistent([state])
                        batch_cache = persistent
                    logits_row = outputs.logits[:, -1, :]
                else:
                    dense, slots = persistent.dense_active_cache(
                        [state.req.request_id for state in decode_states]
                    )
                    outputs = self.model(
                        input_ids=input_ids,
                        past_key_values=dense,
                        use_cache=True,
                    )
                    if isinstance(outputs.past_key_values, BlockPagedBatchCache):
                        dense = outputs.past_key_values
                    persistent.write_back_dense(dense, slots)
                    for idx, state in enumerate(decode_states):
                        state.past_key_values = persistent.extract_cache(slots[idx])
                    batch_cache = persistent
                    logits_row = outputs.logits[:, -1, :]

            next_ids = logits_row.argmax(dim=-1).tolist()
            for idx, state in enumerate(decode_states):
                next_id = int(next_ids[idx])
                state.last_token_id = next_id
                assert state.generated_token_ids is not None
                state.generated_token_ids.append(next_id)
                emitted[state.req.request_id] = self.tokenizer.decode([next_id], skip_special_tokens=True)

        return emitted, batch_cache

    def _paged_next_tokens_batch_for_adapter(
        self, states: list[ActiveState], max_tokens: int
    ) -> dict[str, list[str]]:
        """Multi-step paged decode using persistent batch slots across admissions."""
        emitted: dict[str, list[str]] = {state.req.request_id: [] for state in states}
        if max_tokens <= 0 or not states:
            return emitted

        batch_cache: BlockPagedBatchCache | None = getattr(self, "_paged_persistent", None)

        for _step in range(max_tokens):
            just_prefilled: set[str] = set()
            unprefilled = [state for state in states if not state.prompt_prefilled]
            if unprefilled:
                for rid, token in self._paged_prefill_groups(unprefilled).items():
                    emitted[rid].append(token)
                    just_prefilled.add(rid)
                prefilled = [s for s in states if s.prompt_prefilled]
                if prefilled:
                    batch_cache = self._bind_states_to_persistent(prefilled)

            decode_states = [
                state
                for state in states
                if state.prompt_prefilled and state.req.request_id not in just_prefilled
            ]
            if not decode_states:
                continue

            step_emitted, batch_cache = self._paged_decode_step(decode_states, batch_cache)
            for rid, token in step_emitted.items():
                emitted[rid].append(token)

        return emitted

    def _paged_next_tokens_batch(
        self, states: list[ActiveState], max_tokens: int
    ) -> dict[str, list[str]]:
        emitted: dict[str, list[str]] = {state.req.request_id: [] for state in states}
        by_adapter: dict[str, list[ActiveState]] = collections.defaultdict(list)
        for state in states:
            by_adapter[state.req.adapter or "base"].append(state)
        for adapter, adapter_states in by_adapter.items():
            self.set_active_adapter(adapter)
            partial = self._paged_next_tokens_batch_for_adapter(adapter_states, max_tokens)
            for rid, tokens in partial.items():
                emitted[rid].extend(tokens)
        return emitted

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
            emitted: dict[str, str] = {}
            for state in states:
                self.set_active_adapter(state.req.adapter)
                emitted[state.req.request_id] = self.next_token(state)
            return emitted

        emitted: dict[str, str] = {}
        by_adapter: dict[str, list[ActiveState]] = collections.defaultdict(list)
        for state in states:
            by_adapter[state.req.adapter or "base"].append(state)
        for adapter, adapter_states in by_adapter.items():
            self.set_active_adapter(adapter)
            emitted.update(self._next_token_batch_for_adapter(adapter_states))
        return emitted

    def _next_token_batch_for_adapter(self, states: list[ActiveState]) -> dict[str, str]:
        if self.kv_backend == "paged":
            step_lists = self._paged_next_tokens_batch_for_adapter(states, 1)
            return {rid: tokens[0] for rid, tokens in step_lists.items() if tokens}

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
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True,
                )
                past_values = self._split_past_key_values(outputs.past_key_values, len(group))
                logits = outputs.logits[:, -1, :]
                for idx, state in enumerate(group):
                    state.past_key_values = past_values[idx]
                    state.prompt_prefilled = True
                    next_id = int(logits.argmax(dim=-1).tolist()[idx])
                    state.last_token_id = next_id
                    assert state.generated_token_ids is not None
                    state.generated_token_ids.append(next_id)
                    emitted[state.req.request_id] = self.tokenizer.decode([next_id], skip_special_tokens=True)

            for group in decode_groups.values():
                input_ids = self.torch.tensor([[state.last_token_id] for state in group], device=self.device)
                past = self._concat_past_key_values(group)
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past,
                    use_cache=True,
                )
                past_values = self._split_past_key_values(outputs.past_key_values, len(group))
                logits = outputs.logits[:, -1, :]
                for idx, state in enumerate(group):
                    state.past_key_values = past_values[idx]
                    next_id = int(logits.argmax(dim=-1).tolist()[idx])
                    state.last_token_id = next_id
                    assert state.generated_token_ids is not None
                    state.generated_token_ids.append(next_id)
                    emitted[state.req.request_id] = self.tokenizer.decode([next_id], skip_special_tokens=True)

        return emitted

    def next_tokens_batch(self, states: list[ActiveState], max_tokens: int) -> dict[str, list[str]]:
        if self.kv_backend == "paged":
            return self._paged_next_tokens_batch(states, max_tokens)
        if self.speculative or max_tokens <= 1:
            return {rid: [token] for rid, token in self.next_token_batch(states).items()}
        if not states:
            return {}

        # Keep the model KV state alive. This is intentionally iterative: the old
        # generate() path rebuilt padded prefixes and discarded the cache each call.
        result: dict[str, list[str]] = {state.req.request_id: [] for state in states}
        for _ in range(max_tokens):
            step = self.next_token_batch(states)
            if not step:
                break
            for rid, token in step.items():
                result[rid].append(token)
        return result

    def next_token(self, state: ActiveState) -> str:
        self.set_active_adapter(state.req.adapter)
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
            "quant_mode": self.quant_mode,
            "speculative_enabled": self.speculative,
            "draft_model_id": self.draft_model_id,
            "speculative_k": self.spec_k if self.speculative else 0,
            "speculative_proposed_tokens": self.spec_proposed,
            "speculative_accepted_tokens": self.spec_accepted,
            "speculative_acceptance_rate": acceptance_rate,
            "kv_backend": self.kv_backend,
            **(self.lora_manager.stats() if self.lora_manager is not None else {"lora_enabled": False}),
        }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _allocator, _batch_task, _prefix_cache
    total_blocks = int(os.getenv("KV_TOTAL_BLOCKS", "4096"))
    block_size_tokens = int(os.getenv("KV_BLOCK_SIZE_TOKENS", "16"))
    _allocator = PagedKVAllocator(total_blocks=total_blocks, block_size_tokens=block_size_tokens)
    _prefix_cache = None
    if os.getenv("PHASE2_PREFIX_CACHE", "0") == "1":

        def _retain_blocks(block_ids: list[int]) -> None:
            assert _allocator is not None
            _allocator.retain_blocks(block_ids)

        def _release_blocks(block_ids: list[int]) -> None:
            assert _allocator is not None
            _allocator.release_blocks(block_ids)

        _prefix_cache = PrefixBlockCache(
            block_size_tokens=block_size_tokens,
            max_entries=int(os.getenv("PREFIX_CACHE_MAX_ENTRIES", "256")),
            on_retain_blocks=_retain_blocks,
            on_release_blocks=_release_blocks,
        )
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


def _prune_prefill_handoffs(now: float | None = None) -> None:
    now = now or time.time()
    expired = [hid for hid, payload in _prefill_handoffs.items() if now - float(payload.get("ts", now)) > _prefill_handoff_ttl_s]
    for hid in expired:
        _prefill_handoffs.pop(hid, None)
    while len(_prefill_handoffs) > _prefill_handoff_max:
        _prefill_handoffs.popitem(last=False)


def _prune_terminal_cache(now: float | None = None) -> None:
    now = now or time.time()
    expired = [rid for rid, payload in _terminal_cache.items() if now - payload[0] > _terminal_cache_ttl_s]
    for rid in expired:
        _terminal_cache.pop(rid, None)
    while len(_terminal_cache) > _terminal_cache_max:
        _terminal_cache.popitem(last=False)


def _cache_terminal(request_id: str, payload: dict[str, Any]) -> None:
    _terminal_cache[request_id] = (time.time(), payload)
    _terminal_cache.move_to_end(request_id)
    _prune_terminal_cache()


def _get_cached_terminal(request_id: str) -> dict[str, Any] | None:
    _prune_terminal_cache()
    cached = _terminal_cache.get(request_id)
    if cached is None:
        return None
    _terminal_cache.move_to_end(request_id)
    return cached[1]


def _finalize_request(rid: str, state: ActiveState, reason: str, payload: dict[str, Any], stream_event: dict[str, Any]) -> None:
    if not state.fut.done():
        state.fut.set_result(payload)
    if state.stream_queue is not None:
        state.stream_queue.put_nowait(stream_event)
    if reason == "complete":
        _scheduler_metrics["stream_completed_total"] += 1
    elif reason == "cancel":
        _scheduler_metrics["stream_cancelled_total"] += 1
    elif reason == "timeout":
        _scheduler_metrics["stream_timed_out_total"] += 1
    elif reason == "error":
        _scheduler_metrics["request_errors_total"] += 1
    _cache_terminal(rid, payload)


def _is_enqueued(request_id: str) -> bool:
    return request_id in _queued_request_ids or request_id in _active


def _build_pending(req: GenerateRequest) -> Pending:
    effective_prompt = req.prompt
    if req.prefill_handoff_id:
        _prune_prefill_handoffs()
        handoff = _prefill_handoffs.pop(req.prefill_handoff_id, None)
        if handoff is not None:
            effective_prompt = str(handoff.get("prompt", effective_prompt))
    effective_req = GenerateRequest(
        request_id=req.request_id,
        prompt=effective_prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        deadline_unix_ms=req.deadline_unix_ms,
        prefill_handoff_id=None,
    )
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    return Pending(req=effective_req, fut=fut)


def _try_insert_prefix_cache(state: ActiveState) -> None:
    global _prefix_cache, _allocator
    if _prefix_cache is None or _allocator is None or not state.should_insert_prefix:
        return
    if not state.prompt_prefilled or state.past_key_values is None or state.input_ids is None:
        return
    alloc = _allocator.get_request_allocation(state.req.request_id)
    if alloc is None:
        return
    token_ids = [int(x) for x in state.input_ids[0].tolist()]
    block_size = _allocator.block_size_tokens
    n_blocks = min(len(alloc.block_ids), len(token_ids) // max(1, block_size))
    if n_blocks <= 0:
        return
    prefix_len = n_blocks * block_size
    _prefix_cache.insert(
        token_ids[:prefix_len],
        alloc.block_ids[:n_blocks],
        state.past_key_values,
        state.next_logits,
    )
    state.should_insert_prefix = False


def _release_prefix_entry(state: ActiveState) -> None:
    global _prefix_cache
    if _prefix_cache is None or state.prefix_entry_id is None:
        return
    _prefix_cache.release_entry(state.prefix_entry_id)
    state.prefix_entry_id = None


def _build_pending_stream(req: GenerateRequest, stream_queue: asyncio.Queue) -> Pending:
    pending = _build_pending(req)
    pending.stream_queue = stream_queue
    return pending


async def _continuous_batch_loop() -> None:
    # Orca-style idea: schedule at iteration boundaries, admitting new requests continuously.
    max_active = 32
    decode_step_ms = max(0, int(os.getenv("PHASE2_DECODE_STEP_MS", "0")))
    while True:
        # Pull at least one request if system is idle.
        if not _waiting and not _active:
            pending = await _queue.get()
            _queued_request_ids.discard(pending.req.request_id)
            _waiting.append(pending)

        # Non-blocking drain of new arrivals.
        while True:
            try:
                _waiting.append(_queue.get_nowait())
                _queued_request_ids.discard(_waiting[-1].req.request_id)
            except asyncio.QueueEmpty:
                break

        # Admit waiting requests into active decode set.
        while _waiting and len(_active) < max_active:
            pending = _waiting.popleft()
            try:
                backend = _get_backend()
            except Exception as exc:
                _scheduler_metrics["request_errors_total"] += 1
                payload = {"request_id": pending.req.request_id, "text": "", "error": str(exc)}
                if not pending.fut.done():
                    pending.fut.set_result(payload)
                if pending.stream_queue is not None:
                    pending.stream_queue.put_nowait({"type": "error", "error": str(exc)})
                _cache_terminal(pending.req.request_id, payload)
                _queued_request_ids.discard(pending.req.request_id)
                continue
            assert _allocator is not None
            prompt_tokens = max(1, len((pending.req.prompt or "").split()))
            if backend is not None and hasattr(backend, "tokenizer"):
                try:
                    encoded = backend.tokenizer(pending.req.prompt or "hello", return_tensors="pt")
                    prompt_tokens = max(1, int(encoded["input_ids"].shape[1]))
                except Exception:
                    pass
            token_capacity = prompt_tokens + pending.req.max_tokens
            words = pending.req.prompt.split() or ["hello"]
            state = ActiveState(
                req=pending.req,
                fut=pending.fut,
                stream_queue=pending.stream_queue,
                words=words,
                generated=[],
                cursor=0,
                token_capacity=token_capacity,
                should_insert_prefix=_prefix_cache is not None,
            )
            borrowed_block_ids: list[int] = []
            if backend is not None:
                try:
                    backend.init_state(state)
                    if _prefix_cache is not None:
                        lookup = _prefix_cache.lookup(backend.prompt_token_ids(state))
                        if lookup.hit and lookup.entry is not None:
                            borrowed_block_ids = _prefix_cache.retain_entry(lookup.entry)
                            state.prefix_entry_id = lookup.entry.entry_id
                            state.should_insert_prefix = False
                            state.past_key_values = backend.clone_past_key_values(lookup.entry.past_key_values)
                            state.prompt_prefilled = True
                            token_ids = backend.prompt_token_ids(state)
                            if lookup.matched_tokens > 0:
                                state.last_token_id = token_ids[lookup.matched_tokens - 1]
                            if lookup.entry.next_logits is not None:
                                state.next_logits = lookup.entry.next_logits.clone()
                            else:
                                outputs = backend.model(
                                    input_ids=state.input_ids[:, -1:],
                                    past_key_values=state.past_key_values,
                                    use_cache=True,
                                )
                                state.past_key_values = backend._normalize_past_key_values(outputs.past_key_values)
                                state.next_logits = outputs.logits[:, -1, :]
                except Exception as exc:
                    _scheduler_metrics["request_errors_total"] += 1
                    _finalize_request(
                        pending.req.request_id,
                        state,
                        reason="error",
                        payload={"request_id": pending.req.request_id, "text": "", "error": str(exc)},
                        stream_event={"type": "error", "error": str(exc)},
                    )
                    _queued_request_ids.discard(pending.req.request_id)
                    continue

            alloc = _allocator.allocate_for_tokens(
                pending.req.request_id,
                token_capacity,
                borrowed_block_ids=borrowed_block_ids or None,
            )
            if alloc is None:
                _release_prefix_entry(state)
                _waiting.appendleft(pending)
                break

            if backend is not None:
                try:
                    if state.past_key_values is None and alloc is not None:
                        state.past_key_values = backend._create_kv_cache(token_capacity, alloc.block_ids)
                        if backend.kv_backend == "reserved":
                            state.contiguous_kv_hold = backend._allocate_contiguous_hold(token_capacity)
                except Exception as exc:
                    _scheduler_metrics["request_errors_total"] += 1
                    _finalize_request(
                        pending.req.request_id,
                        state,
                        reason="error",
                        payload={"request_id": pending.req.request_id, "text": "", "error": str(exc)},
                        stream_event={"type": "error", "error": str(exc)},
                    )
                    if _allocator is not None:
                        _allocator.free_request(pending.req.request_id, reason="error")
                    _queued_request_ids.discard(pending.req.request_id)
                    continue
            _active[pending.req.request_id] = state
            _scheduler_metrics["peak_active_requests"] = max(_scheduler_metrics["peak_active_requests"], len(_active))

        if not _active:
            await asyncio.sleep(0.001)
            continue

        # One decode iteration: advance every active request by one token.
        await asyncio.sleep(decode_step_ms / 1000.0 if decode_step_ms else 0)
        _scheduler_metrics["decode_iterations_total"] += 1
        now_ms = int(time.time() * 1000)
        finished_ids: list[str] = []
        finished_reasons: dict[str, str] = {}
        ready: list[ActiveState] = []

        for rid, state in list(_active.items()):
            if _is_cancelled(rid):
                _finalize_request(
                    rid,
                    state,
                    reason="cancel",
                    payload={"request_id": rid, "text": "", "cancelled": True},
                    stream_event={"type": "cancelled"},
                )
                finished_ids.append(rid)
                finished_reasons[rid] = "cancel"
                continue

            if state.req.deadline_unix_ms is not None and now_ms > state.req.deadline_unix_ms:
                _finalize_request(
                    rid,
                    state,
                    reason="timeout",
                    payload={"request_id": rid, "text": "", "timed_out": True},
                    stream_event={"type": "timed_out"},
                )
                finished_ids.append(rid)
                finished_reasons[rid] = "timeout"
                continue

            ready.append(state)
        _scheduler_metrics["max_batch_size"] = max(_scheduler_metrics["max_batch_size"], len(ready))

        backend = _get_backend()
        batched_ids: set[str] = set()
        kv_backend = getattr(backend, "kv_backend", "dynamic") if backend is not None else "dynamic"
        if backend is not None and ready and not getattr(backend, "speculative", False) and kv_backend in {"dynamic", "paged"}:
            _scheduler_metrics["continuous_batches_total"] += 1
            _scheduler_metrics["batched_requests_total"] += len(ready)
            try:
                batch_steps = max(1, int(os.getenv("PHASE2_BATCH_DECODE_STEPS", "16")))
                batch_steps = min(batch_steps, *(state.req.max_tokens - len(state.generated) for state in ready))
                if kv_backend == "paged" and _allocator is not None:
                    for state in ready:
                        rid = state.req.request_id
                        needed = backend.occupied_tokens(state) + batch_steps
                        alloc = _allocator.allocate_for_tokens(rid, needed)
                        if alloc is None:
                            raise RuntimeError(f"paged KV allocation failed for {rid} ({needed} tokens)")
                        backend.sync_kv_blocks(state, alloc.block_ids)
                batch_token_lists = backend.next_tokens_batch(ready, batch_steps)
                for state in ready:
                    rid = state.req.request_id
                    batched_ids.add(rid)
                    for token in batch_token_lists[rid]:
                        next_capacity = backend.occupied_tokens(state) + 1 if backend is not None else max(1, len(state.words)) + len(state.generated) + 1
                        alloc = _allocator.allocate_for_tokens(rid, next_capacity) if _allocator is not None else True
                        if alloc is None:
                            break
                        if backend is not None and alloc is not None:
                            backend.sync_kv_blocks(state, alloc.block_ids)
                        state.generated.append(token)
                        state.cursor += 1
                        if state.stream_queue is not None:
                            state.stream_queue.put_nowait({"type": "token", "token": token, "text": token, "index": state.cursor - 1})
                    if len(state.generated) >= state.req.max_tokens:
                        text = " ".join(state.generated)
                        _finalize_request(
                            rid,
                            state,
                            reason="complete",
                            payload={"request_id": rid, "text": text, "cancelled": False},
                            stream_event={"type": "done", "text": text},
                        )
                        finished_ids.append(rid)
                        finished_reasons[rid] = "complete"
            except Exception as exc:
                for state in ready:
                    rid = state.req.request_id
                    if rid in finished_ids:
                        continue
                    _finalize_request(
                        rid,
                        state,
                        reason="error",
                        payload={"request_id": rid, "text": "", "error": str(exc)},
                        stream_event={"type": "error", "error": str(exc)},
                    )
                    finished_ids.append(rid)
                    finished_reasons[rid] = "error"

        for state in ready:
            rid = state.req.request_id
            if rid in finished_ids or rid in batched_ids:
                continue
            try:
                backend = _get_backend()
                remaining = state.req.max_tokens - len(state.generated)
                tokens = backend.next_tokens(state, remaining) if backend is not None else [state.words[state.cursor % len(state.words)]]
                for token in tokens[:remaining]:
                    next_capacity = backend.occupied_tokens(state) + 1 if backend is not None else max(1, len(state.words)) + len(state.generated) + 1
                    alloc = _allocator.allocate_for_tokens(rid, next_capacity) if _allocator is not None else True
                    if alloc is None:
                        break
                    if backend is not None and alloc is not None:
                        backend.sync_kv_blocks(state, alloc.block_ids)
                    state.generated.append(token)
                    state.cursor += 1
                    if state.stream_queue is not None:
                        state.stream_queue.put_nowait({"type": "token", "token": token, "text": token, "index": state.cursor - 1})
            except Exception as exc:
                _finalize_request(
                    rid,
                    state,
                    reason="error",
                    payload={"request_id": rid, "text": "", "error": str(exc)},
                    stream_event={"type": "error", "error": str(exc)},
                )
                finished_ids.append(rid)
                finished_reasons[rid] = "error"
                continue

            if len(state.generated) >= state.req.max_tokens:
                text = " ".join(state.generated)
                _finalize_request(
                    rid,
                    state,
                    reason="complete",
                    payload={"request_id": rid, "text": text, "cancelled": False},
                    stream_event={"type": "done", "text": text},
                )
                finished_ids.append(rid)
                finished_reasons[rid] = "complete"

        _record_cuda_peak()

        for rid in finished_ids:
            state = _active.pop(rid, None)
            if state is not None:
                _release_prefix_entry(state)
                state.contiguous_kv_hold = None
            backend = _backend
            if backend is not None and hasattr(backend, "release_paged_request"):
                backend.release_paged_request(rid)
            if _allocator is not None:
                _allocator.free_request(rid, reason=finished_reasons.get(rid, "complete"))
            _cancelled.pop(rid, None)
            _queued_request_ids.discard(rid)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    cached = _get_cached_terminal(req.request_id)
    if cached is not None:
        return cached
    if _is_enqueued(req.request_id):
        return {"request_id": req.request_id, "text": "", "duplicate": True}
    pending = _build_pending(req)
    _queued_request_ids.add(req.request_id)
    await _queue.put(pending)
    return await pending.fut


@app.post("/prefill")
async def prefill(req: PrefillRequest) -> dict[str, Any]:
    handoff_id = f"{req.request_id}-prefill-{int(time.time() * 1e6)}"
    _prefill_handoffs[handoff_id] = {
        "ts": time.time(),
        "request_id": req.request_id,
        "prompt": req.prompt,
    }
    _prefill_handoffs.move_to_end(handoff_id)
    _prune_prefill_handoffs()
    return {
        "request_id": req.request_id,
        "prefill_handoff_id": handoff_id,
        "prefill_tokens": max(1, len(req.prompt.split())),
    }


@app.post("/decode")
async def decode(req: DecodeRequest) -> dict[str, Any]:
    cached = _get_cached_terminal(req.request_id)
    if cached is not None:
        return cached
    if _is_enqueued(req.request_id):
        return {"request_id": req.request_id, "text": "", "duplicate": True}
    pending = _build_pending(
        GenerateRequest(
            request_id=req.request_id,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            deadline_unix_ms=req.deadline_unix_ms,
            prefill_handoff_id=req.prefill_handoff_id,
        )
    )
    _queued_request_ids.add(req.request_id)
    await _queue.put(pending)
    return await pending.fut


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    cached = _get_cached_terminal(req.request_id)
    if cached is not None:
        async def replay_events() -> Any:
            if cached.get("cancelled"):
                yield json.dumps({"type": "cancelled"}) + "\n"
            elif cached.get("timed_out"):
                yield json.dumps({"type": "timed_out"}) + "\n"
            elif cached.get("error"):
                yield json.dumps({"type": "error", "error": cached.get("error")}) + "\n"
            else:
                for idx, token in enumerate(str(cached.get("text", "")).split()):
                    yield json.dumps({"type": "token", "token": token, "text": token, "index": idx}) + "\n"
                yield json.dumps({"type": "done", "text": cached.get("text", "")}) + "\n"

        return StreamingResponse(replay_events(), media_type="application/x-ndjson")

    if _is_enqueued(req.request_id):
        async def duplicate_events() -> Any:
            yield json.dumps({"type": "duplicate"}) + "\n"

        return StreamingResponse(duplicate_events(), media_type="application/x-ndjson")

    stream_queue: asyncio.Queue = asyncio.Queue()
    pending = _build_pending_stream(req, stream_queue)
    _queued_request_ids.add(req.request_id)
    await _queue.put(pending)
    fut = pending.fut

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
        "active_requests": len(_active),
        "active_allocations_current": _allocator.stats().get("active_allocations", 0) if _allocator is not None else 0,
    }
    base.update(_scheduler_metrics)
    if _allocator is not None:
        base.update(_allocator.stats())
    if _backend is not None:
        base.update(_backend.metrics())
    if _prefix_cache is not None:
        base.update(_prefix_cache.stats())
    base["peak_torch_cuda_bytes"] = _peak_torch_cuda_bytes
    backend = _backend
    if backend is not None and getattr(backend, "kv_pool", None) is not None:
        base["paged_kv_pool_peak_bytes"] = int(backend.kv_pool.peak_allocated_bytes())
        base["paged_kv_pool_allocated_blocks"] = int(backend.kv_pool.allocated_block_count())
    return base
