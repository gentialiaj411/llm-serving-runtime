# ADR-0004: Model Parallelism Strategy

- Status: Proposed
- Date: 2026-05-15

## Context
Need early direction before multi-GPU (Phase 5).

## Decision
Default to request-level replication across workers. Revisit TP/PP before Phase 5 with benchmark-driven trigger criteria.

## Consequences
Simplifies early reliability/scheduling work while preserving later scaling options.
