from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class Allocation:
    request_id: str
    block_ids: list[int]
    requested_tokens: int


class PagedKVAllocator:
    """Simple paged KV allocator with fragmentation accounting.

    - Fixed-size blocks
    - First-fit from free list
    - Per-request block ownership tracking
    """

    def __init__(self, total_blocks: int, block_size_tokens: int, bytes_per_token: int = 1) -> None:
        if total_blocks <= 0:
            raise ValueError("total_blocks must be > 0")
        if block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be > 0")
        if bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be > 0")

        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens
        self.bytes_per_token = bytes_per_token
        self.block_size_bytes = block_size_tokens * bytes_per_token
        self._free: list[int] = list(range(total_blocks))
        self._owned: dict[str, Allocation] = {}

    def allocate_for_tokens(self, request_id: str, token_capacity: int) -> Allocation | None:
        needed = max(1, math.ceil(token_capacity / self.block_size_tokens))
        if request_id in self._owned:
            return self._owned[request_id]
        if needed > len(self._free):
            return None

        ids = [self._free.pop() for _ in range(needed)]
        alloc = Allocation(request_id=request_id, block_ids=ids, requested_tokens=token_capacity)
        self._owned[request_id] = alloc
        return alloc

    def free_request(self, request_id: str) -> None:
        alloc = self._owned.pop(request_id, None)
        if alloc is None:
            return
        self._free.extend(alloc.block_ids)

    def get_request_allocation(self, request_id: str) -> Allocation | None:
        return self._owned.get(request_id)

    def stats(self) -> dict[str, float | int]:
        used_blocks = self.total_blocks - len(self._free)
        used_pct = (used_blocks / self.total_blocks) * 100.0

        # External fragmentation proxy: free blocks are present but many requests can't fit
        # contiguous in a real allocator. In this list-based allocator, we model pressure as
        # ratio of free blocks to largest single request-free capacity.
        largest_alloc = 0
        for alloc in self._owned.values():
            largest_alloc = max(largest_alloc, len(alloc.block_ids))
        free_blocks = len(self._free)
        frag_ratio = 0.0
        if free_blocks > 0 and largest_alloc > 0:
            frag_ratio = max(0.0, 1.0 - (free_blocks / (free_blocks + largest_alloc)))

        return {
            "total_blocks": self.total_blocks,
            "free_blocks": free_blocks,
            "used_blocks": used_blocks,
            "used_pct": used_pct,
            "block_size_tokens": self.block_size_tokens,
            "bytes_per_token": self.bytes_per_token,
            "block_size_bytes": self.block_size_bytes,
            "total_kv_bytes": self.total_blocks * self.block_size_bytes,
            "free_kv_bytes": free_blocks * self.block_size_bytes,
            "active_allocations": len(self._owned),
            "fragmentation_ratio": frag_ratio,
        }
