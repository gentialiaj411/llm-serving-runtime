# ADR-0002: KV Cache Interface Contract

- Status: Proposed
- Date: 2026-05-15

## Context
Paged attention arrives in Phase 4, but API must be stable earlier.

## Decision
Define block-oriented KV interface now (`allocate_blocks`, `append_tokens`, `free_blocks`, metrics hooks).

## Consequences
Allows naive backend in Phase 2 while preserving migration path to paged allocator.
