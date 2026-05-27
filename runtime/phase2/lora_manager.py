"""PEFT-based multi-LoRA hot-swap on a shared base model."""

from __future__ import annotations

import json
import os
from typing import Any


def _target_modules_for_config(config: Any) -> list[str]:
    model_type = getattr(config, "model_type", "").lower()
    if "qwen" in model_type or "llama" in model_type or "mistral" in model_type:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "gpt2" in model_type:
        return ["c_attn", "c_proj"]
    return ["q_proj", "v_proj"]


class LoRAManager:
    """Registers multiple LoRA adapters and hot-swaps without reloading the base weights."""

    def __init__(
        self,
        base_model: Any,
        *,
        enabled: bool,
        adapter_names: list[str],
        adapter_paths: dict[str, str] | None = None,
        rank: int = 8,
        alpha: int = 16,
    ) -> None:
        self.enabled = enabled
        self.adapter_names = adapter_names
        self.adapter_paths = adapter_paths or {}
        self._active = "base"
        self._swap_total = 0
        self.model = base_model

        if not enabled:
            return

        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as exc:
            raise RuntimeError("PHASE2_LORA=1 requires the peft package") from exc

        trainable = [name for name in adapter_names if name != "base"]
        if not trainable:
            trainable = ["adapter_a"]

        target_modules = _target_modules_for_config(base_model.config)
        first = trainable[0]
        first_path = self.adapter_paths.get(first)
        if first_path:
            self.model = PeftModel.from_pretrained(base_model, first_path, adapter_name=first, is_trainable=False)
        else:
            cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=target_modules)
            self.model = get_peft_model(base_model, cfg, adapter_name=first)

        for name in trainable[1:]:
            path = self.adapter_paths.get(name)
            if path:
                self.model.load_adapter(path, adapter_name=name, is_trainable=False)
            else:
                self.model.add_adapter(
                    name,
                    LoraConfig(r=rank, lora_alpha=alpha, target_modules=target_modules),
                )

        self.model.eval()

    def set_active(self, adapter: str) -> None:
        if not self.enabled:
            return
        adapter = adapter or "base"
        if adapter not in self.adapter_names and adapter != "base":
            adapter = "base"
        if adapter == self._active:
            return
        if adapter == "base":
            self.model.disable_adapter()
        else:
            self.model.set_adapter(adapter)
            if hasattr(self.model, "enable_adapter_layers"):
                self.model.enable_adapter_layers()
        self._active = adapter
        self._swap_total += 1

    def stats(self) -> dict[str, Any]:
        return {
            "lora_enabled": self.enabled,
            "lora_adapters": list(self.adapter_names),
            "lora_active_adapter": self._active,
            "lora_adapter_swaps_total": self._swap_total,
        }


def load_adapter_config_from_env() -> tuple[list[str], dict[str, str]]:
    names_raw = os.getenv("LORA_ADAPTER_NAMES", "base,adapter_a,adapter_b,adapter_c")
    names = [n.strip() for n in names_raw.split(",") if n.strip()]
    paths: dict[str, str] = {}
    paths_raw = os.getenv("LORA_ADAPTER_PATHS_JSON", "").strip()
    if paths_raw:
        parsed = json.loads(paths_raw)
        if isinstance(parsed, dict):
            paths = {str(k): str(v) for k, v in parsed.items()}
    return names, paths
