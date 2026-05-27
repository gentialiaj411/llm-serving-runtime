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
        self._block_refs: dict[int, int] = {}
        self._owned: dict[str, Allocation] = {}
        self._allocations_total = 0
        self._allocation_failures_total = 0
        self._frees_total = 0
        self._frees_cancel_total = 0
        self._frees_timeout_total = 0
        self._frees_error_total = 0
        self._peak_used_blocks = 0
        self._peak_active_allocations = 0

    def retain_blocks(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            self._block_refs[block_id] = self._block_refs.get(block_id, 0) + 1
            if block_id in self._free:
                self._free.remove(block_id)

    def release_blocks(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            refs = self._block_refs.get(block_id, 0)
            if refs <= 1:
                self._block_refs.pop(block_id, None)
                if block_id not in self._free and block_id < self.total_blocks:
                    self._free.append(block_id)
            else:
                self._block_refs[block_id] = refs - 1

    def allocate_for_tokens(
        self,
        request_id: str,
        token_capacity: int,
        *,
        borrowed_block_ids: list[int] | None = None,
    ) -> Allocation | None:
        needed = max(1, math.ceil(token_capacity / self.block_size_tokens))
        if request_id in self._owned:
            existing = self._owned[request_id]
            have = len(existing.block_ids)
            if needed <= have:
                return existing
            extra_needed = needed - have
            if extra_needed > len(self._free):
                self._allocation_failures_total += 1
                return None
            extra_ids = [self._free.pop() for _ in range(extra_needed)]
            existing.block_ids.extend(extra_ids)
            existing.requested_tokens = max(existing.requested_tokens, token_capacity)
            self._allocations_total += 1
            self._update_peaks()
            return existing
        borrowed = list(borrowed_block_ids or [])
        if len(borrowed) > needed:
            borrowed = borrowed[:needed]
        extra_needed = needed - len(borrowed)
        if extra_needed > len(self._free):
            self._allocation_failures_total += 1
            return None

        ids = borrowed + [self._free.pop() for _ in range(extra_needed)]
        for block_id in borrowed:
            self.retain_blocks([block_id])
        alloc = Allocation(request_id=request_id, block_ids=ids, requested_tokens=token_capacity)
        self._owned[request_id] = alloc
        self._allocations_total += 1
        self._update_peaks()
        return alloc

    def free_request(self, request_id: str, reason: str = "complete") -> bool:
        alloc = self._owned.pop(request_id, None)
        if alloc is None:
            return False
        for block_id in alloc.block_ids:
            refs = self._block_refs.get(block_id, 0)
            if refs <= 1:
                self._block_refs.pop(block_id, None)
                self._free.append(block_id)
            else:
                self._block_refs[block_id] = refs - 1
        self._frees_total += 1
        if reason == "cancel":
            self._frees_cancel_total += 1
        elif reason == "timeout":
            self._frees_timeout_total += 1
        elif reason == "error":
            self._frees_error_total += 1
        return True

    def get_request_allocation(self, request_id: str) -> Allocation | None:
        return self._owned.get(request_id)

    def _update_peaks(self) -> None:
        used_blocks = self.total_blocks - len(self._free)
        self._peak_used_blocks = max(self._peak_used_blocks, used_blocks)
        self._peak_active_allocations = max(self._peak_active_allocations, len(self._owned))

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
            "used_kv_bytes": used_blocks * self.block_size_bytes,
            "active_allocations": len(self._owned),
            "peak_used_blocks": self._peak_used_blocks,
            "peak_kv_bytes": self._peak_used_blocks * self.block_size_bytes,
            "peak_active_allocations": self._peak_active_allocations,
            "allocations_total": self._allocations_total,
            "allocation_failures_total": self._allocation_failures_total,
            "frees_total": self._frees_total,
            "frees_cancel_total": self._frees_cancel_total,
            "frees_timeout_total": self._frees_timeout_total,
            "frees_error_total": self._frees_error_total,
            "fragmentation_ratio": frag_ratio,
        }
