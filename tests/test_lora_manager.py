from __future__ import annotations

import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from runtime.phase2.lora_manager import LoRAManager, load_adapter_config_from_env


class LoRAManagerTests(unittest.TestCase):
    def test_registers_three_adapters(self) -> None:
        config = GPT2Config(vocab_size=128, n_layer=2, n_head=2, n_embd=32)
        base = GPT2LMHeadModel(config)
        base.eval()
        names, _ = load_adapter_config_from_env()
        mgr = LoRAManager(base, enabled=True, adapter_names=names[:4], adapter_paths={})
        self.assertIn("adapter_a", mgr.adapter_names)
        mgr.set_active("adapter_a")
        mgr.set_active("adapter_b")
        self.assertGreaterEqual(mgr.stats()["lora_adapter_swaps_total"], 1)


if __name__ == "__main__":
    unittest.main()
