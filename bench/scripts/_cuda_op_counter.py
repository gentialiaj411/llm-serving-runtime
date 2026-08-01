"""Count GPU-dispatching aten ops as a CUPTI-free launch proxy."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _is_cuda_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor) and x.is_cuda


def _tree_has_cuda(obj: Any) -> bool:
    if _is_cuda_tensor(obj):
        return True
    if isinstance(obj, (tuple, list)):
        return any(_tree_has_cuda(x) for x in obj)
    if isinstance(obj, dict):
        return any(_tree_has_cuda(v) for v in obj.values())
    return False


class CudaOpCounter(TorchDispatchMode):
    """Counts aten ops that touch CUDA tensors (proxy for kernel launches)."""

    def __init__(self) -> None:
        super().__init__()
        self.total = 0
        self.by_op: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if _tree_has_cuda(args) or _tree_has_cuda(kwargs) or _tree_has_cuda(out):
            name = str(func)
            self.total += 1
            self.by_op[name] += 1
        return out
